"""Tests for Project.generate_project_table and load_project_table.

The project table stacks every recording's features on one comparable timeline and
attaches cohort metadata, so results can be grouped by a recorded attribute across
recordings. Cohorts differ between recordings, so the animal_id and phase enums have to
be widened to strings before the concat, and recordings of different lengths stay ragged
rather than being padded or truncated.
"""

import datetime as dt
import json
import warnings
from pathlib import Path

import polars as pl
import pytest
import strategies

from deepecohab.app import services
from deepecohab.core.data_model import Bout, Event, Layout, Project, Recording

TZ = "UTC"
# Animals shuttle between neighbouring cages, which is enough to populate every table.
ROUTE = [1, 2, 3, 2]


def raw_reads(recording: Recording, hours: int) -> pl.DataFrame:
	"""Antenna registrations for every animal, walking the route once per hour.

	Each animal is offset by a few minutes so they overlap in cages without being
	perfectly synchronised, which gives the pairwise and chasing steps something to find.
	"""
	origin = recording.timeline.experiment_start
	rows = []

	for offset, animal in enumerate(recording.cohort.animal_tags):
		moment = origin + dt.timedelta(minutes=3 * offset)
		for hour in range(hours):
			for step, antenna in enumerate(ROUTE):
				rows.append(
					{
						"datetime": moment
						+ dt.timedelta(hours=hour, minutes=7 * step, seconds=offset),
						"antenna": antenna,
						"time_under": dt.timedelta(milliseconds=120),
						"animal_id": animal,
						"inserted": False,
					}
				)

	return pl.DataFrame(rows, schema=recording.data_schema).sort("datetime")


def write_recording(directory: Path, recording: Recording, hours: int) -> tuple[Path, Path, Path]:
	"""Persist a recording's files the way Project.add_recording reads them.

	The diagnostic is the minimal one: no recording boundaries, nothing interpolated.
	"""
	directory.mkdir(parents=True, exist_ok=True)
	metadata_path = directory / f"{recording.name}.config.json"
	data_path = directory / f"{recording.name}.data.parquet"
	diagnostic_path = directory / f"{recording.name}.diagnostic.json"

	metadata_path.write_text(
		json.dumps({"recording": recording.model_dump(mode="json")}, indent=2), encoding="utf-8"
	)
	raw_reads(recording, hours).write_parquet(data_path)
	diagnostic_path.write_text(strategies.MINIMAL_DIAGNOSTIC, encoding="utf-8")
	return metadata_path, data_path, diagnostic_path


def make_recording(name: str, animal_ids: list[str], genotype: str, finish: str) -> Recording:
	recording = strategies.analysis_recording(
		animal_ids=animal_ids, tz=TZ, start="2023-05-24 00:00:00", finish=finish
	)
	recording.name = name
	for animal in recording.cohort.animals:
		animal.genotype = genotype
	return recording


@pytest.fixture(scope="module")
def project(tmp_path_factory) -> Project:
	"""A two-recording project, analysed, with different cohorts and different lengths.

	The cohorts deliberately share no tags, so a concat that kept the per-recording
	animal_id enum would fail here rather than in production.
	"""
	root = tmp_path_factory.mktemp("project_table")
	sources = []

	for name, animals, genotype, finish, hours in (
		("wt_cohort", ["A", "B", "C"], "WT", "2023-05-27 00:00:00", 60),
		("ko_cohort", ["X", "Y"], "KO", "2023-05-26 00:00:00", 36),
	):
		recording = make_recording(name, animals, genotype, finish)
		sources.extend(write_recording(root / "sources" / name, recording, hours))

	project = Project.create(
		project_name="table_test", experimenter="tester", location=root / "project"
	)
	project.add_recordings(sources)
	project.run_analysis()
	return project


def test_covers_every_recording_and_animal(project):
	table = project.generate_project_table().collect()

	assert set(table["recording"].unique()) == {"wt_cohort", "ko_cohort"}
	assert set(table["animal_id"].unique()) == {"A", "B", "C", "X", "Y"}


def test_row_count_is_the_sum_of_the_parts(project):
	"""Nothing is dropped or duplicated by the concat."""
	table = project.generate_project_table().collect()
	parts = sum(
		recording.load_results("feature_df", eager=True).height for recording in project.recordings
	)
	assert table.height == parts


def test_enums_are_widened_to_strings(project):
	"""Per-recording enum categories cannot survive a concat across cohorts."""
	table = project.generate_project_table().collect()

	assert table.schema["animal_id"] == pl.String
	assert table.schema["phase"] == pl.String
	assert table.schema["position"] == pl.String


def test_position_type_names_what_each_position_is(project):
	"""Cages carry their cage type, tunnels "tunnel" and unplaced time stays undefined."""
	table = project.generate_project_table().collect()
	layout = project.recordings[0].layout
	types = dict(table.select("position", "position_type").unique().iter_rows())

	assert {types[cage] for cage in layout.cage_names} == {"standard"}
	assert {types[tunnel] for tunnel in layout.tunnel_names} == {"tunnel"}
	assert types[Layout.UNDEFINED] == Layout.UNDEFINED
	assert types[None] is None  # the chasing metrics, which belong to no position


def test_metadata_travels_with_every_row(project):
	"""The point of the table: grouping by a recorded attribute spans recordings."""
	table = project.generate_project_table().collect()

	assert table["genotype"].null_count() == 0
	by_genotype = table.group_by("genotype").agg(pl.col("animal_id").n_unique().alias("animals"))
	assert dict(by_genotype.iter_rows()) == {"WT": 3, "KO": 2}


def test_cohort_size_is_carried_so_the_correction_can_be_undone(project):
	table = project.generate_project_table().collect()
	sizes = table.group_by("recording").agg(pl.col("n_mice").unique().alias("sizes"))

	assert {row["recording"]: row["sizes"] for row in sizes.iter_rows(named=True)} == {
		"wt_cohort": [3],
		"ko_cohort": [2],
	}


def test_recordings_stay_ragged(project):
	"""A shorter recording contributes the days it has, with nothing padded in."""
	table = project.generate_project_table().collect()
	last_day = table.group_by("recording").agg(pl.max("day").alias("last"))
	days = {row["recording"]: row["last"] for row in last_day.iter_rows(named=True)}

	assert days["wt_cohort"] > days["ko_cohort"]


def test_rates_are_comparable_across_recordings(project):
	"""Both halves survive the concat, so a rate is a sum-then-divide over the table."""
	table = project.generate_project_table().collect()
	rates = (
		table.filter(pl.col("metric") == "activity")
		.group_by("genotype")
		.agg(pl.sum("value"), pl.sum("exposure"))
		.with_columns((pl.col("value") / pl.col("exposure")).alias("visits_per_hour"))
	)

	assert rates.height == 2
	assert (rates["visits_per_hour"] > 0).all()


def test_load_returns_what_generate_wrote(project):
	written = project.generate_project_table().collect()

	assert project.load_project_table(eager=True).equals(written)
	assert project.load_project_table().collect().equals(written)
	assert (project.project_location / Project.PROJECT_TABLE).is_file()


def test_load_before_generate_raises(tmp_path):
	empty = Project.create(project_name="empty", experimenter="tester", location=tmp_path / "p")
	with pytest.raises(FileNotFoundError, match="generate_project_table"):
		empty.load_project_table()


def test_generate_with_no_recordings_raises(tmp_path):
	empty = Project.create(project_name="empty", experimenter="tester", location=tmp_path / "p")
	with pytest.raises(ValueError, match="no recordings"):
		empty.generate_project_table()


def test_event_columns_are_added_and_null_for_recordings_that_declare_nothing(tmp_path_factory):
	"""One column per project event, plus Any event; null throughout for a recording
	that never declares the event, "Other hours" for one that declares it but this
	cell is neither during a bout nor at the same hour of another day.
	"""
	root = tmp_path_factory.mktemp("project_table_events")

	with_event = make_recording("with_event", ["A", "B"], "WT", "2023-05-27 00:00:00")
	with_event.events = [
		Event(
			name="Tone",
			description="a tone",
			bouts=[
				Bout(
					start=dt.datetime(2023, 5, 24, 5, 0, tzinfo=dt.UTC),
					end=dt.datetime(2023, 5, 24, 5, 10, tzinfo=dt.UTC),
				)
			],
		)
	]
	without_event = make_recording("without_event", ["X", "Y"], "KO", "2023-05-26 00:00:00")

	sources = [
		*write_recording(root / "sources" / "with_event", with_event, 60),
		*write_recording(root / "sources" / "without_event", without_event, 48),
	]

	project = Project.create(
		project_name="events", experimenter="tester", location=root / "project"
	)
	project.add_recordings(sources)
	project.run_analysis()

	table = project.generate_project_table().collect()

	assert "Tone" in table.columns
	assert "Any event" in table.columns

	declaring = table.filter(pl.col("recording") == "with_event")
	assert declaring.filter((pl.col("day") == 1) & (pl.col("hour") == 5))[
		"Tone"
	].unique().to_list() == ["During"]
	assert declaring.filter((pl.col("day") == 2) & (pl.col("hour") == 5))[
		"Tone"
	].unique().to_list() == ["Same hours, other days"]
	assert declaring.filter((pl.col("day") == 1) & (pl.col("hour") == 6))[
		"Tone"
	].unique().to_list() == ["Other hours"]
	assert declaring["Any event"].null_count() == 0

	silent = table.filter(pl.col("recording") == "without_event")
	assert silent["Tone"].null_count() == silent.height
	assert silent["Any event"].null_count() == silent.height


def test_names_selects_a_subset(project):
	table = project.generate_project_table(names=["wt_cohort"]).collect()
	assert set(table["recording"].unique()) == {"wt_cohort"}


def test_removing_a_recording_drops_it_from_the_project_table(tmp_path):
	sources = []
	for name, animals, genotype, finish, hours in (
		("keep", ["A", "B"], "WT", "2023-05-26 00:00:00", 36),
		("drop", ["X", "Y"], "KO", "2023-05-26 00:00:00", 36),
	):
		recording = make_recording(name, animals, genotype, finish)
		sources.extend(write_recording(tmp_path / "sources" / name, recording, hours))

	project = Project.create(
		project_name="remove_test", experimenter="tester", location=tmp_path / "project"
	)
	project.add_recordings(sources)
	project.run_analysis()
	project.generate_project_table()

	project.remove_recording("drop")

	table = project.load_project_table(eager=True)
	assert set(table["recording"].unique()) == {"keep"}


def test_a_delisted_recording_can_be_reinstated_after_a_reload(tmp_path):
	recording = make_recording("back", ["A", "B"], "WT", "2023-05-26 00:00:00")
	files = write_recording(tmp_path / "src", recording, 12)
	project = Project.create(
		project_name="reinstate", experimenter="tester", location=tmp_path / "project"
	)
	project.add_recording(*files)
	project.remove_recording("back")
	project.close()

	with Project.load(project.project_location) as reopened:
		assert reopened.delisted == {"back": "back/config.json"}
		reopened.reinstate_recording("back")

	with Project.load(project.project_location) as reopened:
		assert list(reopened.data_catalog) == ["back"]
		assert reopened.delisted == {}


def test_deleting_a_recording_leaves_nothing_to_reinstate(tmp_path):
	recording = make_recording("gone", ["A", "B"], "WT", "2023-05-26 00:00:00")
	files = write_recording(tmp_path / "src", recording, 12)
	project = Project.create(
		project_name="delete", experimenter="tester", location=tmp_path / "project"
	)
	project.add_recording(*files)
	project.remove_recording("gone", delete_files=True)

	with pytest.raises(KeyError, match="No delisted recording"):
		project.reinstate_recording("gone")


def test_re_adding_a_recording_replaces_it_and_its_results(tmp_path):
	project = Project.create(
		project_name="replace", experimenter="tester", location=tmp_path / "project"
	)
	first = make_recording("same", ["A", "B"], "WT", "2023-05-26 00:00:00")
	project.add_recording(*write_recording(tmp_path / "v1", first, 12))
	project.run_analysis()
	project.generate_project_table()

	second = make_recording("same", ["A", "B"], "KO", "2023-05-26 00:00:00")
	files = write_recording(tmp_path / "v2", second, 12)
	with pytest.raises(FileExistsError):
		project.add_recording(*files)
	with pytest.warns(UserWarning, match="left as they are"):
		report = project.add_recordings(files)
	assert list(report.existing) == ["same"]
	assert {a.genotype for a in project["same"].cohort.animals} == {"WT"}

	with pytest.warns(UserWarning, match="Replaced recording 'same'"):
		project.add_recording(*files, overwrite=True)

	assert list(project.data_catalog) == ["same"]
	assert {a.genotype for a in project["same"].cohort.animals} == {"KO"}
	assert not any(project["same"].results_path.iterdir())
	assert not (project.project_location / Project.PROJECT_TABLE).is_file()
	assert sorted(p.name for p in project.project_location.iterdir() if p.is_dir()) == ["same"]


def test_removing_the_last_recording_deletes_the_project_table(tmp_path):
	recording = make_recording("only", ["A", "B"], "WT", "2023-05-26 00:00:00")
	files = write_recording(tmp_path / "src", recording, 12)

	project = Project.create(
		project_name="remove_all", experimenter="tester", location=tmp_path / "project"
	)
	project.add_recording(*files)
	project.run_analysis()
	project.generate_project_table()

	project.remove_recording("only")

	assert not (project.project_location / Project.PROJECT_TABLE).is_file()


def test_adding_a_trimmed_recording_warns_with_the_duration(tmp_path):
	"""Data dropped ahead of the experiment start is reported once, on add."""
	recording = strategies.analysis_recording(
		tz=TZ,
		start="2023-05-24 09:30:00",
		finish="2023-05-27 07:00:00",
		phases={"light_phase": dt.time(7, 0), "dark_phase": dt.time(20, 0)},
		start_from="dark_phase",
		end_with="dark_phase",
	)
	recording.name = "late_start"
	files = write_recording(tmp_path / "src", recording, 12)

	project = Project.create(
		project_name="warns", experimenter="tester", location=tmp_path / "project"
	)
	with pytest.warns(UserWarning, match="10:30:00 of data recorded before it is left out"):
		project.add_recording(*files)


def test_adding_a_late_started_recording_reports_the_short_first_phase(tmp_path):
	"""The other direction: recording began after the onset, so nothing is dropped.

	The nearest onset is the one just behind, which keeps the day that skipping forward
	would have cost, at the price of a slightly short first phase.
	"""
	recording = strategies.analysis_recording(
		tz=TZ,
		# The lead has to clear LEAD_WARNING_THRESHOLD for the warning to be raised at all.
		start="2023-05-24 21:10:00",
		finish="2023-05-27 07:00:00",
		phases={"light_phase": dt.time(7, 0), "dark_phase": dt.time(20, 0)},
		start_from="dark_phase",
		end_with="dark_phase",
	)
	recording.name = "just_missed"
	files = write_recording(tmp_path / "src", recording, 12)

	assert recording.timeline.discarded_lead == dt.timedelta(0)
	project = Project.create(
		project_name="gains", experimenter="tester", location=tmp_path / "project"
	)
	with pytest.warns(UserWarning, match="started 1:10:00 into it.*no data is dropped"):
		project.add_recording(*files)


@pytest.mark.parametrize(
	("start", "name"),
	[
		("2023-05-24 20:07:30", "late_by_minutes"),  # a short unrecorded lead
		("2023-05-24 19:52:30", "early_by_minutes"),  # a short discarded lead
	],
)
def test_a_lead_under_the_threshold_is_quiet(tmp_path, start, name):
	"""Acquisition is rarely started on the onset, so a few minutes either way is routine.

	The offset is still logged; it just does not warn, which is what keeps the warning
	worth reading on the recordings that are genuinely far off.
	"""
	recording = strategies.analysis_recording(
		tz=TZ,
		start=start,
		finish="2023-05-27 07:00:00",
		phases={"light_phase": dt.time(7, 0), "dark_phase": dt.time(20, 0)},
		start_from="dark_phase",
		end_with="dark_phase",
	)
	recording.name = name
	files = write_recording(tmp_path / "src", recording, 12)

	line = recording.timeline
	assert (
		dt.timedelta(0) < (line.discarded_lead or line.unrecorded_lead) < dt.timedelta(minutes=30)
	)

	project = Project.create(
		project_name="quiet_lead", experimenter="tester", location=tmp_path / name
	)
	with warnings.catch_warnings():
		warnings.simplefilter("error")
		project.add_recording(*files)


def test_adding_an_untrimmed_recording_is_quiet(tmp_path):
	recording = strategies.analysis_recording(
		tz=TZ,
		start="2023-05-24 20:00:00",
		finish="2023-05-27 07:00:00",
		phases={"light_phase": dt.time(7, 0), "dark_phase": dt.time(20, 0)},
		start_from="dark_phase",
		end_with="dark_phase",
	)
	recording.name = "on_time"
	files = write_recording(tmp_path / "src", recording, 12)

	project = Project.create(
		project_name="quiet", experimenter="tester", location=tmp_path / "project"
	)
	with warnings.catch_warnings():
		warnings.simplefilter("error")
		project.add_recording(*files)


def test_files_group_into_recordings_by_name(tmp_path):
	recording = make_recording("grouped", ["A", "B"], "WT", "2023-05-26 00:00:00")
	config, data, diagnostic = write_recording(tmp_path / "src", recording, 12)
	orphan = tmp_path / "src" / "orphan.config.json"
	orphan.write_text("{}", encoding="utf-8")
	stray = tmp_path / "src" / "notes.txt"
	stray.write_text("", encoding="utf-8")

	project = Project.create(
		project_name="grouped", experimenter="tester", location=tmp_path / "project"
	)
	with pytest.raises(
		ValueError,
		match=r"orphan: no data file or diagnostic file - expected orphan\.data\.parquet",
	):
		project.add_recording(orphan)
	with pytest.warns(UserWarning, match="2 of 3"):
		report = project.add_recordings([stray, data, orphan, diagnostic, config])

	assert report.added == ["grouped"]
	assert {f.name: type(f.error) for f in report.failed} == {
		"notes.txt": ValueError,
		"orphan": FileNotFoundError,
	}
	raw = project["grouped"].root / "raw"
	assert (raw / "diagnostic.json").read_bytes() == diagnostic.read_bytes()
	assert (raw / "config.json").read_bytes() == config.read_bytes()


def test_update_notes_persists_to_config_json(tmp_path):
	recording = strategies.analysis_recording(
		tz=TZ, start="2023-05-24 00:00:00", finish="2023-05-25 00:00:00"
	)
	recording.name = "noted"
	files = write_recording(tmp_path / "src", recording, 6)

	project = Project.create(
		project_name="notes", experimenter="tester", location=tmp_path / "project"
	)
	project.add_recording(*files)
	project.update_notes("noted", "checked the water bottles twice a day")

	reloaded = Project.load(project.project_location)
	assert reloaded["noted"].notes == "checked the water bottles twice a day"


def test_update_notes_leaves_a_line_in_the_project_log(tmp_path):
	recording = strategies.analysis_recording(
		tz=TZ, start="2023-05-24 00:00:00", finish="2023-05-25 00:00:00"
	)
	recording.name = "noted"
	files = write_recording(tmp_path / "src", recording, 6)

	project = Project.create(
		project_name="notes", experimenter="tester", location=tmp_path / "project"
	)
	project.add_recording(*files)
	project.update_notes("noted", "water bottles checked", tag="A")

	log = (project.project_location / Project.LOGFILE).read_text(encoding="utf-8")
	assert "noted: notes updated for animal A" in log


@pytest.mark.parametrize("manifest", [None, "not json"])
def test_project_name_falls_back_to_the_folder_name(tmp_path, manifest):
	if manifest is not None:
		(tmp_path / Project.MANIFEST).write_text(manifest, encoding="utf-8")
	assert services.project_name(str(tmp_path)) == tmp_path.name


def test_project_name_reads_the_manifest(tmp_path):
	project = Project.create(
		project_name="named", experimenter="tester", location=tmp_path / "project"
	)
	assert services.project_name(str(project.project_location)) == "named"


def test_a_project_that_fails_to_load_is_summarised_from_its_manifest(tmp_path, monkeypatch):
	monkeypatch.setattr(services, "CACHE_DIR", tmp_path / "cache")
	recording = make_recording("broken", ["A", "B"], "WT", "2023-05-25 00:00:00")
	project = Project.create(
		project_name="summary", experimenter="tester", location=tmp_path / "project"
	)
	project.add_recording(*write_recording(tmp_path / "src", recording, 6))
	(project.project_location / "broken" / Project.CONFIG).write_text("{}", encoding="utf-8")

	summary = services.project_summary(str(project.project_location))

	assert summary["error"]
	assert (summary["name"], summary["experimenter"]) == ("summary", "tester")
	assert [row["name"] for row in summary["recordings"]] == ["broken"]


def test_update_notes_with_a_tag_persists_to_one_animal(tmp_path):
	"""A tag targets that animal's notes and leaves the recording's own notes alone."""
	recording = strategies.analysis_recording(
		tz=TZ, start="2023-05-24 00:00:00", finish="2023-05-25 00:00:00"
	)
	recording.name = "noted"
	files = write_recording(tmp_path / "src", recording, 6)

	project = Project.create(
		project_name="notes", experimenter="tester", location=tmp_path / "project"
	)
	project.add_recording(*files)
	project.update_notes("noted", "skittish since the cage change", tag="A")

	reloaded = Project.load(project.project_location)
	by_tag = {animal.tag: animal for animal in reloaded["noted"].cohort.animals}
	assert by_tag["A"].notes == "skittish since the cage change"
	assert by_tag["B"].notes == ""
	assert reloaded["noted"].notes == ""
