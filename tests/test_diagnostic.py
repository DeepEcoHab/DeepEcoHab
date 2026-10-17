"""Tests for recording boundaries and interpolated reads from the acquisition software.

Run on a synthetic Europe/Warsaw recording whose reads stop from 03:20 to 07:00: its
``diagnostic.json`` holds that stop as one boundary, in UTC, and every fifth read is
flagged ``inserted``. Each animal's last read before the stop is at antenna 3 and its
first after at antenna 1, a step that would cross antenna 2 unread were it not a stop.
"""

import copy
import datetime as dt
import json

import polars as pl
import pytest
import strategies
from test_project_table import make_recording, raw_reads, write_recording

from deepecohab.core.data_model import AnalysisParams, Layout, Project, Recording
from deepecohab.core.recording_pipeline import build_main_df, build_recording_quality

NAME = "stopped"


def stopped_files(directory):
	recording = strategies.analysis_recording(
		animal_ids=["A", "B"],
		tz="Europe/Warsaw",
		start="2023-05-24 00:00:00",
		finish="2023-05-25 00:00:00",
	)
	recording.name = NAME
	config, data, diagnostic = write_recording(directory, recording, 24)

	origin = recording.timeline.experiment_start
	reads = raw_reads(recording, 24).with_row_index()
	gap = pl.col("datetime").is_between(
		origin + dt.timedelta(hours=3, minutes=20), origin + dt.timedelta(hours=7), closed="left"
	)
	reads = reads.filter(~gap).with_columns(inserted=pl.col("index") % 5 == 0).drop("index")
	reads.write_parquet(data)

	before = reads.filter(pl.col("datetime") < origin + dt.timedelta(hours=7))["datetime"].max()
	after = reads.filter(pl.col("datetime") >= origin + dt.timedelta(hours=7))["datetime"].min()
	boundary = {
		"start": before.astimezone(dt.UTC).isoformat(),
		"end": after.astimezone(dt.UTC).isoformat(),
		"kind": "observed_record_gap",
	}
	diagnostic.write_text(json.dumps({"recording_boundaries": [boundary]}), encoding="utf-8")
	return config, data, diagnostic


@pytest.fixture(scope="module")
def recording(tmp_path_factory) -> Recording:
	root = tmp_path_factory.mktemp("diagnostic")
	project = Project.create(project_name="diagnostic", experimenter="tester", location=root / "p")
	project.add_recording(*stopped_files(root / "src"))
	project.close()
	return Project.load(project.project_location)[NAME]


def test_boundaries_are_read_in_the_recording_zone(recording):
	"""Stored in config.json at add, and back after a reload in the recording's own zone."""
	assert [boundary.kind for boundary in recording.boundaries] == ["observed_record_gap"]
	assert all(
		boundary.start.tzinfo == recording.timeline.recording_timezone
		for boundary in recording.boundaries
	)


def test_every_animal_is_undefined_from_the_stop_to_its_first_read_after_it(recording):
	"""Carried up to the stop, then unknown until each animal is read again."""
	stop = recording.boundaries[0]
	main = build_main_df(recording, AnalysisParams()).collect()

	inside = main.filter((pl.col("datetime") > stop.start) & (pl.col("datetime") < stop.end))
	at_stop = main.filter(pl.col("datetime") == stop.start)
	first_after = (
		main.filter(pl.col("datetime") >= stop.end)
		.group_by("animal_id")
		.agg(pl.all().sort_by("datetime").first())
	)

	assert inside.is_empty()
	assert at_stop.height == recording.cohort.n_mice
	assert (at_stop["position"] != Layout.UNDEFINED).all()
	assert (first_after["position"] == Layout.UNDEFINED).all()
	assert (first_after["datetime"] - first_after["time_spent"] == stop.start).all()


def test_quality_counts_only_the_analysed_window(recording):
	"""Correct and interpolated add up to the reads in the window; bad never spans a stop."""
	start, end = recording.timeline.local_span
	in_window = recording.data.filter(pl.col("datetime").is_between(start, end)).collect()

	quality = build_recording_quality(recording, AnalysisParams()).collect()
	unbounded = build_recording_quality(
		copy.replace(recording, boundaries=[]), AnalysisParams()
	).collect()

	assert in_window["inserted"].any()
	assert quality["correct"].sum() == (~in_window["inserted"]).sum()
	assert quality["interpolated"].sum() == in_window["inserted"].sum()
	assert quality["bad"].sum() == 0
	assert unbounded["bad"].sum() == recording.cohort.n_mice
	assert quality["miss_rate"].is_between(0, 100).all()


def test_a_recording_without_a_stored_diagnostic_fails_to_load(tmp_path):
	recording = make_recording("old", ["A", "B"], "WT", "2023-05-26 00:00:00")
	project = Project.create(project_name="t", experimenter="t", location=tmp_path / "project")
	project.add_recording(*write_recording(tmp_path / "src", recording, 6))
	project.close()
	(project["old"].root / "raw" / "diagnostic.json").unlink()

	with pytest.raises(FileNotFoundError, match=r"'old' has no raw/diagnostic\.json"):
		Project.load(project.project_location)
