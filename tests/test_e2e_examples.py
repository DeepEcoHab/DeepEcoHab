"""End-to-end test of the whole pipeline on the bundled example recordings.

Drives the public API exactly as ``examples/new_backend_prototype.ipynb`` does -
create a project, add every recording in ``examples/data``, run every registered
step, sink the auxiliary tube test, aggregate the project table, then build every
registered plot - so it guards the create -> analyse -> plot path on real data.
The focused unit tests feed synthetic frames one step at a time and cannot see
the relationships *between* the produced tables, which is what is asserted here.

The example data ships with the repository, so this test never skips: if the
data is gone, that is the failure.
"""

import io
import json
import zipfile
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlencode

import plotly.graph_objects as go
import polars as pl
import pytest
from dash import Dash, html
from plotly.io.json import to_json_plotly

import deepecohab as deh
from deepecohab.app import downloads, services
from deepecohab.app.builder import figure as builder_figure, presets
from deepecohab.app.components import export_preview
from deepecohab.auxiliary_analysis.tube_test import calculate_tube_test
from deepecohab.core.data_model import DataFrameRegistry
from deepecohab.plotting import PlotContext, PlotRegistry

pytestmark = pytest.mark.e2e

DATA_DIR = Path(__file__).resolve().parent.parent / "examples" / "data"
STEPS = DataFrameRegistry.step_order()
PLOTS = PlotRegistry.list_available()
#: The export dialog's defaults, as its render callback hands them to export_preview.
EXPORT_FORM = {
	"style": "publication",
	"format": "svg",
	"width_mm": 85.0,
	"height_mm": 64.0,
	"pt": 8.0,
	"dpi": 300,
	"legend": True,
	"title_on": False,
	"events": True,
	"csv": True,
}


@pytest.fixture(scope="session")
def project(tmp_path_factory) -> Iterator[deh.Project]:
	"""A project over every example recording, analysed end to end.

	Session-scoped: the pipeline runs once for the whole module.
	"""
	configs = list(DATA_DIR.glob("*.config.json"))
	if not configs:
		raise FileNotFoundError(f"No example recordings in {DATA_DIR}; the e2e test needs them.")

	created = deh.Project.create(
		project_name="e2e_examples",
		experimenter="Tester",
		location=tmp_path_factory.mktemp("e2e") / "project",
	)
	report = created.add_recordings(DATA_DIR.iterdir())
	assert not report.failed, f"recordings failed to add: {report.failed}"
	assert len(report.added) == len(configs)

	created.run_analysis()
	# Not a pipeline step: sinking it is the caller's call, and the tube-test plot
	# needs it on disk.
	for recording in created.recordings:
		calculate_tube_test(recording, max_dwell=2.0, winner_behavior="BOTH").sink_parquet(
			recording.results_path / "tube_test_df.parquet",
			compression="lz4",
			engine="streaming",
		)
	created.generate_project_table()
	created.close()

	# Reopened from the manifest, so every test below reads the persisted project
	# rather than the in-memory one that wrote it.
	reloaded = deh.Project.load(created.project_location)
	yield reloaded
	reloaded.close()


@pytest.fixture(scope="session")
def context(project) -> PlotContext:
	"""One recording's plotting context, caching the tables it has read."""
	return PlotContext.from_recording(project.recordings[0])


def _tables(recording: deh.Recording) -> dict[str, pl.DataFrame]:
	"""Every table the recording produced, including the auxiliary tube test."""
	return {key: recording.load_results(key, eager=True) for key in [*STEPS, "tube_test_df"]}


# --- project + pipeline ------------------------------------------------------


def test_reload_finds_every_recording(project):
	"""The manifest round-trips: every recording reopens with its metadata attached."""
	assert len(project) == len(list(DATA_DIR.glob("*.config.json")))
	for recording in project.recordings:
		assert recording.name == recording.root.name
		assert recording.cohort.animal_tags
		assert recording.layout.cage_names


@pytest.mark.parametrize("step", STEPS)
def test_step_produced_a_nonempty_table(project, step):
	"""Every registered step ran for every recording and wrote a non-empty parquet."""
	for recording in project.recordings:
		assert deh.recording_status(recording.root)[step], f"{recording.name}: {step} not built"
		assert recording.load_results(step).select(pl.len()).collect().item() > 0


def test_rerunning_the_analysis_skips_finished_steps(project):
	"""A second run resumes rather than redoing: no parquet is rewritten."""
	before = {
		path: path.stat().st_mtime_ns
		for recording in project.recordings
		for path in recording.results_path.glob("*.parquet")
	}
	project.run_analysis()

	assert {path: path.stat().st_mtime_ns for path in before} == before


def test_no_nan_or_inf_in_any_float_column(project):
	"""No table carries a null, NaN or inf in a float column.

	The grids are zero-filled and the rate metrics guard their denominators; a stray
	NaN would otherwise flow silently into the plots and downstream stats.
	"""
	for recording in project.recordings:
		for key, table in _tables(recording).items():
			for column, dtype in table.schema.items():
				if dtype not in (pl.Float32, pl.Float64):
					continue
				values = table[column]
				where = f"{recording.name}.{key}.{column}"
				assert values.null_count() == 0, f"{where} has nulls"
				assert not values.is_nan().any(), f"{where} has NaN"
				assert not values.is_infinite().any(), f"{where} has inf"


# --- cross-table consistency on real data ------------------------------------


def test_chasings_grid_reconciles_with_match_events(project):
	"""The per-hour chasings grid sums to exactly the chasing events it came from.

	chasings_df is match_df aggregated and zero-filled onto the dense grid, so the
	totals must agree per chaser - no event lost or invented by the expansion.
	"""
	for recording in project.recordings:
		matches = recording.load_results("match_df", eager=True)
		chasings = recording.load_results("chasings_df", eager=True)

		assert chasings.filter(pl.col("chaser") == pl.col("chased")).height == 0
		assert int(chasings["chasings"].sum()) == matches.height

		per_chaser = {
			chaser: int(count)
			for chaser, count in chasings.group_by("chaser").agg(pl.sum("chasings")).iter_rows()
			if count > 0
		}
		expected = {
			winner: int(count)
			for winner, count in matches.group_by("winner").agg(pl.len()).iter_rows()
		}
		assert per_chaser == expected


def test_time_alone_never_exceeds_occupancy(project):
	"""Solitary time cannot exceed total occupancy of the same animal/position cell.

	Both sides are binned with ``get_grid_phase_count``, so the two joins in
	``calculate_activity`` line up cell for cell; a misalignment drops an occupancy
	row while keeping its solitary-time counterpart.
	"""
	for recording in project.recordings:
		activity = recording.load_results("activity_df", eager=True)
		offenders = activity.filter(pl.col("time_alone") > pl.col("time_in_position"))

		assert offenders.height == 0, f"{recording.name}:\n{offenders}"


def test_hourly_durations_fit_inside_their_hour(project):
	"""No hourly cell holds more than an hour of occupancy or togetherness.

	The padded frame attributes time to the hour it elapsed in, so a cell over 3600s
	means time was double-counted or a span leaked across a boundary.
	"""
	for recording in project.recordings:
		activity = recording.load_results("activity_df", eager=True)
		pairwise = recording.load_results("pairwise_meetings", eager=True)

		assert activity.filter(pl.col("time_in_position") > pl.duration(hours=1)).height == 0
		assert activity.filter(pl.col("time_alone") > pl.duration(hours=1)).height == 0
		assert pairwise.filter(pl.col("time_together") > pl.duration(hours=1)).height == 0


def test_pairs_are_stored_unordered(project):
	"""Every pair table keys on ``animal_id < animal_id_2``, so no pair is counted twice."""
	for recording in project.recordings:
		for key in ("pairwise_meetings", "incohort_sociability"):
			table = recording.load_results(key, eager=True)
			ordered = table["animal_id"].cast(pl.String) < table["animal_id_2"].cast(pl.String)

			assert ordered.all(), f"{recording.name}.{key} has unordered pairs"


def test_encounters_never_exceed_the_pairs_visits(project):
	"""A pair's encounters in a cell cannot outnumber the two animals' visits to it.

	Every encounter starts with one of the pair arriving, and distinct encounters have
	distinct arrivals, so the count is bounded by the pair's own visits. Not bounded by
	either animal alone: one can sit still while the other comes and goes. Regression
	guard for the encounter inflation that float-second rounding used to cause in the
	reconstructed visit starts.
	"""
	cell = ["day", "phase", "hour", "phase_count", "position"]

	for recording in project.recordings:
		visits = recording.load_results("activity_df").select(
			*cell, "animal_id", "visits_to_position"
		)
		pairs = (
			recording.load_results("pairwise_meetings")
			.join(visits, on=[*cell, "animal_id"], how="left")
			.join(
				visits.rename({"animal_id": "animal_id_2", "visits_to_position": "visits_2"}),
				on=[*cell, "animal_id_2"],
				how="left",
			)
		)
		offenders = pairs.filter(
			pl.col("pairwise_encounters") > pl.col("visits_to_position") + pl.col("visits_2")
		).collect()

		assert offenders.height == 0, f"{recording.name}:\n{offenders}"


def test_sociability_proportion_stays_within_cage_bounds(project):
	"""proportion_together sums a per-cage fraction, so it lies in [0, n_cages]."""
	for recording in project.recordings:
		sociability = recording.load_results("incohort_sociability", eager=True)
		proportion = sociability["proportion_together"]

		assert proportion.min() >= 0
		assert proportion.max() <= len(recording.layout.cage_names)


def test_tube_test_scores_real_pairs(project):
	"""Tube-test counts are non-negative, self-pairs absent, animals in the cohort."""
	for recording in project.recordings:
		tube = recording.load_results("tube_test_df", eager=True)
		cohort = set(recording.cohort.animal_tags)

		assert set(tube["winner"].cast(pl.String).unique()) <= cohort
		assert set(tube["loser"].cast(pl.String).unique()) <= cohort
		assert tube.filter(pl.col("winner") == pl.col("loser")).height == 0
		assert (tube["tube_test"] >= 0).all()


def test_every_cohort_animal_reaches_the_feature_table(project):
	"""No animal drops out of the ML feature set, however inactive it was."""
	for recording in project.recordings:
		features = recording.load_results("feature_df", eager=True)

		assert set(features["animal_id"].cast(pl.String).unique()) == set(
			recording.cohort.animal_tags
		)


def test_project_table_aggregates_every_recording(project):
	"""The project table carries each recording's features, cohort metadata and events."""
	table = project.load_project_table(eager=True)
	declared = sorted(
		{event.name for recording in project.recordings for event in recording.events}
	)

	assert set(table["recording"].unique()) == set(project.data_catalog)
	assert {"recording", "n_mice", "metric", "value", "exposure", "genotype"} <= set(table.columns)
	assert {*declared, "Any event"} <= set(table.columns)
	assert set(table["Any event"].unique()) <= {
		"During",
		"Same hours, other days",
		"Other hours",
	}
	assert table["day"].min() >= 1


# --- plots -------------------------------------------------------------------


def assert_plotly_accepts(figure: dict, label: str) -> None:
	"""plotly takes every key of ``figure`` and changes none of its values.

	A plot is built as a plain dict, so this is the validation plotly's own figure objects
	would have run: an unknown key raises, and a key plotly.js would not read - an
	underscored shorthand such as ``line_width`` - comes back rewritten.
	"""

	# Rounded: plotly's angle validator wraps an angle, which moves its last digits.
	def read(payload: str) -> dict:
		return json.loads(payload, parse_float=lambda text: round(float(text), 9))

	sent = read(to_json_plotly(figure))
	accepted = read(go.Figure(figure).to_json())
	accepted["layout"].pop("template", None)

	assert accepted == sent, label


@pytest.mark.parametrize("plot_name", PLOTS)
def test_plot_builds_for_every_option_value(context, plot_name):
	"""Every registered plot builds on real tables, for each value of each option, into a
	figure plotly accepts as it is."""
	assert_plotly_accepts(deh.plot(plot_name, context), plot_name)

	for option in PlotRegistry.spec(plot_name).options_for(context):
		for choice in option.choices:
			# A multi-select option takes a list; a single-select takes the value.
			value = [choice] if isinstance(option.default, (list, tuple)) else choice
			figure = deh.plot(plot_name, context, **{option.name: value})

			assert_plotly_accepts(figure, f"{plot_name}[{option.name}={choice}]")


@pytest.fixture(scope="module")
def export_client():
	"""The export route on a Dash server, whose request hooks run as they do in the app.

	A Dash app also has to exist for export_preview's warning icons to resolve their assets.
	"""
	app = Dash(__name__)
	app.layout = html.Div()  # Dash's first-request hook refuses to serve without one
	app.server.register_blueprint(downloads.bp)
	return app.server.test_client()


@pytest.mark.parametrize("plot_name", PLOTS)
def test_plot_downloads_from_the_export_dialog(project, context, export_client, plot_name):
	"""Every registered plot downloads with the dialog's defaults, posted as the dialog posts
	it, with the frames it was drawn from as CSV beside the image."""
	form = {**EXPORT_FORM, "filename": plot_name}
	# JSON round-tripped, as dcc.Graph hands the figure over: no numpy arrays or datetimes.
	figure = json.loads(to_json_plotly(deh.plot(plot_name, context)))
	# As deh.plotRequest sends it for a card left at its defaults.
	request = {
		"context": {
			"location": str(project.project_location),
			"recording": project.recordings[0].name,
		},
		"controls": {},
		"opts": {},
	}
	payload = export_preview(figure, plot_name, form, {"plot": plot_name, "request": request})

	# Encoded as the browser encodes a form, escaping every JSON delimiter: Werkzeug's test
	# client leaves some unescaped and so posts a smaller body than the dialog does.
	body = urlencode({"figure": payload["payload_figure"], "params": payload["payload_params"]})
	resp = export_client.post(
		"/export", data=body, content_type="application/x-www-form-urlencoded"
	)

	assert resp.status_code == 200, resp.get_data(as_text=True)[:200]
	frames = PlotRegistry.data(plot_name, context)
	with zipfile.ZipFile(io.BytesIO(resp.data)) as zf:
		names = zf.namelist()
		csvs = {name: pl.read_csv(zf.read(name)) for name in names if name.endswith(".csv")}
	assert f"{plot_name}.svg" in names
	assert [csv.height for csv in csvs.values()] == [frame.height for frame in frames.values()]


def test_builder_plot_downloads_its_drawn_rows(project, export_client):
	"""A builder plot's CSV is one row per drawn point, outliers flagged rather than dropped."""
	# The time-alone preset, filters in the plain shape the builder page hands figure.py.
	state = {
		**presets.BY_ID["time-alone-genotype-sex"].state,
		"filters": {"metric": ["time_alone"]},
		"outliers": "exclude",
	}
	frame, fields = services.builder_frame(str(project.project_location))
	figure, _ = builder_figure.build_figure(frame, state, fields)
	data = {"builder": {"location": str(project.project_location), "state": state}}
	payload = export_preview(
		json.loads(figure.to_json()), "Time alone", {**EXPORT_FORM, "filename": "b"}, data
	)

	body = urlencode({"figure": payload["payload_figure"], "params": payload["payload_params"]})
	resp = export_client.post(
		"/export", data=body, content_type="application/x-www-form-urlencoded"
	)

	assert resp.status_code == 200, resp.get_data(as_text=True)[:200]
	with zipfile.ZipFile(io.BytesIO(resp.data)) as zf:
		csv = pl.read_csv(zf.read("b.csv"))
	animals = sum(len(recording.cohort.animals) for recording in project.recordings)
	assert csv.height == animals
	assert {"genotype", "sex", "recording", "animal id", "excluded_outlier"} <= set(csv.columns)
	drawn = sum(len(trace.y) for trace in figure.data)
	assert csv.filter(~pl.col("excluded_outlier")).height == drawn
