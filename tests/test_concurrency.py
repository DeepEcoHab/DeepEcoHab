"""Tests for analysing several recordings at once.

Recordings share nothing - their own results/ directory, a step graph that is read-only
once imported - so their steps, and a recording's independent steps, run concurrently. These tests
pin what that must not cost: the same tables as a serial run, every step still produced and
reported exactly once, and a failure that neither hides nor strands the other recordings.
"""

import threading
from pathlib import Path

import pytest
from polars.testing import assert_frame_equal
from test_project_table import make_recording, write_recording

from deepecohab.core.data_model import DataFrameRegistry, Project

RECORDINGS = (
	("wt_cohort", ["A", "B", "C"], "WT", "2023-05-27 00:00:00", 60),
	("ko_cohort", ["X", "Y"], "KO", "2023-05-26 00:00:00", 36),
	("het_cohort", ["P", "Q"], "HET", "2023-05-26 00:00:00", 36),
)


@pytest.fixture(scope="module")
def sources(tmp_path_factory) -> list[Path]:
	"""Three recordings on disk, ready for Project.add_recordings."""
	root = tmp_path_factory.mktemp("concurrency_sources")
	return [
		path
		for name, animals, genotype, finish, hours in RECORDINGS
		for path in write_recording(root, make_recording(name, animals, genotype, finish), hours)
	]


def build_project(location, sources, **kwargs) -> Project:
	"""A fresh project holding the same three recordings, analysed."""
	project = Project.create(project_name="concurrency", experimenter="tester", location=location)
	project.add_recordings(sources)
	project.run_analysis(**kwargs)
	return project


def test_concurrent_matches_serial(tmp_path, sources):
	"""The tables do not depend on how many recordings were in flight.

	Compared without row order, and without ``row_id``: polars' lazy engine does not
	promise either, sinks least of all, and two serial runs already disagree on both
	wherever rows tie on ``datetime``. Neither carries meaning across runs, so pinning
	them would only test the engine's scheduling. The content is what has to match.
	"""
	serial = build_project(tmp_path / "serial", sources, workers=1)
	concurrent = build_project(tmp_path / "concurrent", sources, workers=3)

	for name, *_ in RECORDINGS:
		for key in DataFrameRegistry.list_available():
			expected = serial[name].load_results(key, eager=True)
			produced = concurrent[name].load_results(key, eager=True)
			label = ["row_id"] if "row_id" in expected.columns else []
			assert_frame_equal(produced.drop(label), expected.drop(label), check_row_order=False)

	serial.close()
	concurrent.close()


def test_every_table_lands(tmp_path, sources):
	"""Concurrency must not drop a step: every recording gets every parquet."""
	project = build_project(tmp_path / "complete", sources, workers=3)

	for name, *_ in RECORDINGS:
		produced = {path.stem for path in project[name].results_path.glob("*.parquet")}
		assert produced == set(DataFrameRegistry.list_available())

	project.close()


def test_event_stream_covers_every_step(tmp_path, sources):
	"""Every (recording, step) pair is reported exactly once, however they interleave."""
	project = Project.create(
		project_name="concurrency", experimenter="tester", location=tmp_path / "events"
	)
	project.add_recordings(sources)

	events = list(project.iter_analysis(workers=3))
	steps = DataFrameRegistry.step_order()

	assert len(events) == len(RECORDINGS) * len(steps)
	assert {(event.recording, event.step) for event in events} == {
		(name, step) for name, *_ in RECORDINGS for step in steps
	}
	# Within one recording a step never arrives before a step it requires, and the
	# index counts that recording's finished steps.
	for name, *_ in RECORDINGS:
		arrived = [e.step for e in events if e.recording == name]
		for position, step in enumerate(arrived):
			assert set(DataFrameRegistry._requires[step]) <= set(arrived[:position])
		assert [e.step_index for e in events if e.recording == name] == list(
			range(1, len(steps) + 1)
		)

	project.close()


def test_failure_surfaces_without_stranding_the_others(tmp_path, sources, monkeypatch):
	"""One bad recording raises, and the recordings that were fine still finish."""
	doomed = RECORDINGS[0][0]
	original = DataFrameRegistry._builders["animals"]

	def explode(recording, params):
		if recording.name == doomed:
			raise RuntimeError("boom")
		return original(recording, params)

	monkeypatch.setitem(DataFrameRegistry._builders, "animals", explode)

	project = Project.create(
		project_name="concurrency", experimenter="tester", location=tmp_path / "failure"
	)
	project.add_recordings(sources)

	with pytest.raises(RuntimeError, match="boom"):
		project.run_analysis(workers=3)

	assert not (project[doomed].results_path / "animals.parquet").is_file()
	for name, *_ in RECORDINGS[1:]:
		produced = {path.stem for path in project[name].results_path.glob("*.parquet")}
		assert produced == set(DataFrameRegistry.list_available())

	# The caller shows one line of the error; the traceback has to survive in the log.
	log = (project.project_location / Project.LOGFILE).read_text(encoding="utf-8")
	assert "analysis failed" in log
	assert "RuntimeError: boom" in log

	project.close()


def test_cancel_lets_the_step_in_flight_land_and_starts_nothing_new(tmp_path, sources, monkeypatch):
	"""Cancelling mid-step finishes that step; no later step or recording starts."""
	cancel = threading.Event()
	first = DataFrameRegistry.step_order()[0]
	original = DataFrameRegistry._builders[first]

	def cancelled_while_building(recording, params):
		cancel.set()
		return original(recording, params)

	monkeypatch.setitem(DataFrameRegistry._builders, first, cancelled_while_building)

	project = Project.create(
		project_name="concurrency", experimenter="tester", location=tmp_path / "cancel"
	)
	project.add_recordings(sources)

	events = list(project.iter_analysis(workers=1, cancel=cancel))

	assert [(event.recording, event.step) for event in events] == [(RECORDINGS[0][0], first)]
	produced = {
		(name, path.stem)
		for name, *_ in RECORDINGS
		for path in project[name].results_path.glob("*.parquet")
	}
	assert produced == {(RECORDINGS[0][0], first)}

	project.close()
