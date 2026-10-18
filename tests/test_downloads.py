"""Tests for the download and export Flask routes.

A tiny analysed project on disk stands in for a real one; Flask's own test client
drives the blueprint directly. `services.remember_project` is what a normal render
of the Projects page would already have done, so the fixture does the same thing by
hand rather than going through Dash.
"""

import datetime as dt
import io
import json
import zipfile

import plotly.graph_objects as go
import polars as pl
import pytest
import strategies
from flask import Flask

from deepecohab.app import downloads, services
from deepecohab.core.data_model import Project, Recording
from deepecohab.plotting import PlotRegistry

ROUTE = [1, 2, 3, 2]


def _raw_reads(recording: Recording, hours: int) -> pl.DataFrame:
	origin = recording.timeline.experiment_start
	rows = [
		{
			"datetime": origin + dt.timedelta(hours=hour, minutes=7 * step, seconds=offset),
			"antenna": antenna,
			"time_under": dt.timedelta(milliseconds=120),
			"animal_id": animal,
			"inserted": False,
		}
		for offset, animal in enumerate(recording.cohort.animal_tags)
		for hour in range(hours)
		for step, antenna in enumerate(ROUTE)
	]
	return pl.DataFrame(rows, schema=recording.data_schema).sort("datetime")


@pytest.fixture(scope="module")
def project(tmp_path_factory) -> Project:
	root = tmp_path_factory.mktemp("downloads")
	recording = strategies.analysis_recording()
	recording.name = "rec1"

	metadata_path = root / "rec1.config.json"
	data_path = root / "rec1.data.parquet"
	metadata_path.write_text(
		json.dumps({"recording": recording.model_dump(mode="json")}), encoding="utf-8"
	)
	_raw_reads(recording, hours=48).write_parquet(data_path)
	diagnostic_path = root / "rec1.diagnostic.json"
	diagnostic_path.write_text(strategies.MINIMAL_DIAGNOSTIC, encoding="utf-8")

	project = Project.create(
		project_name="dl_test", experimenter="tester", location=root / "project"
	)
	project.add_recording(metadata_path, data_path, diagnostic_path)
	project.run_analysis()
	return project


@pytest.fixture
def client(project, monkeypatch, tmp_path):
	monkeypatch.setattr(services, "CACHE_DIR", tmp_path / "cache")
	app = Flask(__name__)
	app.register_blueprint(downloads.bp)
	pid = services.remember_project(str(project.project_location))
	return app.test_client(), pid


def test_recording_table_streams_a_registered_step(client):
	test_client, pid = client

	resp = test_client.get(f"/download/recording/{pid}/rec1/table/main_df.parquet")

	assert resp.status_code == 200
	assert resp.headers["Content-Disposition"].startswith("attachment")


def test_download_token_comes_back_as_a_cookie_with_the_file(client):
	test_client, pid = client

	resp = test_client.get(f"/download/recording/{pid}/rec1/config.json?download_token=abc")

	assert "deh-download=abc" in resp.headers["Set-Cookie"]


def test_recording_table_404s_for_an_uncomputed_or_unknown_name(client):
	test_client, pid = client

	assert (
		test_client.get(f"/download/recording/{pid}/rec1/table/not_a_table.parquet").status_code
		== 404
	)


def test_unknown_recording_404s(client):
	test_client, pid = client

	assert (
		test_client.get(f"/download/recording/{pid}/not_a_recording/config.json").status_code == 404
	)


def test_recording_tables_zip_bundles_every_computed_step(client):
	test_client, pid = client

	resp = test_client.get(f"/download/recording/{pid}/rec1/tables.zip")

	with zipfile.ZipFile(io.BytesIO(resp.data)) as zf:
		names = zf.namelist()
	assert "main_df.parquet" in names


def test_recording_tables_zip_as_csv_converts_durations_to_seconds(client):
	test_client, pid = client

	resp = test_client.get(f"/download/recording/{pid}/rec1/tables.zip?format=csv")

	with zipfile.ZipFile(io.BytesIO(resp.data)) as zf:
		text = zf.read("main_df.csv").decode()
	header = text.splitlines()[0]
	assert "time_under" in header
	# a duration column in seconds is a plain float, not a microsecond integer or struct
	assert "." in text.splitlines()[1].split(",")[header.split(",").index("time_under")]


def test_recording_cohort_csv_lists_every_animal(client):
	test_client, pid = client

	resp = test_client.get(f"/download/recording/{pid}/rec1/cohort.csv")

	lines = resp.data.decode().splitlines()
	assert "tag" in lines[0]
	assert len(lines) == 4  # header + 3 animals


def test_recording_raw_and_config_stream(client):
	test_client, pid = client

	assert test_client.get(f"/download/recording/{pid}/rec1/raw.parquet").status_code == 200
	assert test_client.get(f"/download/recording/{pid}/rec1/config.json").status_code == 200


def test_unknown_project_id_404s(client):
	test_client, _ = client

	assert test_client.get("/download/project/deadbeefcafe/table").status_code == 404


def test_project_archive_zip_excludes_raw_by_default(client):
	test_client, pid = client

	resp = test_client.get(f"/download/project/{pid}/archive.zip")

	with zipfile.ZipFile(io.BytesIO(resp.data)) as zf:
		names = zf.namelist()
	assert any(name.endswith("config.json") for name in names)
	assert not any("raw" in name for name in names)


def test_project_archive_zip_includes_raw_when_asked(client):
	test_client, pid = client

	resp = test_client.get(f"/download/project/{pid}/archive.zip?raw=1")

	with zipfile.ZipFile(io.BytesIO(resp.data)) as zf:
		assert any("raw" in name for name in zf.namelist())


def _export_params(**overrides) -> dict:
	base = {
		"style": "publication",
		"format": "svg",
		"width_mm": 85,
		"height_mm": 64,
		"pt": 8,
		"dpi": 300,
		"legend": True,
		"title_on": False,
		"events": True,
		"csv": False,
		"title": "",
		"filename": "plot",
	}
	return {**base, **overrides}


def test_export_renders_an_image(client):
	test_client, _ = client
	fig = go.Figure(go.Scatter(x=[1, 2], y=[3, 4])).to_dict()

	resp = test_client.post(
		"/export", data={"figure": json.dumps(fig), "params": json.dumps(_export_params())}
	)

	assert resp.status_code == 200
	assert resp.mimetype == "image/svg+xml"


def test_export_bundles_the_frame_the_plot_was_drawn_from(client, project):
	test_client, _ = client
	fig = go.Figure(go.Scatter(x=[1, 2], y=[3, 4])).to_dict()
	location = str(project.project_location)
	request = {
		"context": {"location": location, "recording": "rec1"},
		"controls": {"hours": [0, 23]},
		"opts": {"metric": "visits"},
	}
	params = _export_params(csv=True, data={"plot": "activity-bar", "request": request})

	resp = test_client.post(
		"/export", data={"figure": json.dumps(fig), "params": json.dumps(params)}
	)

	assert resp.mimetype == "application/zip"
	with zipfile.ZipFile(io.BytesIO(resp.data)) as zf:
		assert set(zf.namelist()) == {"plot.svg", "plot.csv"}
		csv = pl.read_csv(zf.read("plot.csv"))
	(frame,) = PlotRegistry.data(
		"activity-bar", services.plot_context(location, "rec1"), metric="visits"
	).values()
	assert csv.columns == frame.columns
	assert csv.height == frame.height > 0


def test_export_names_one_csv_per_frame(client, monkeypatch):
	test_client, _ = client
	fig = go.Figure(go.Scatter(x=[1, 2], y=[3, 4])).to_dict()
	frames = {"nodes": pl.DataFrame({"a": [1]}), "edges": pl.DataFrame({"b": [2]})}
	monkeypatch.setattr(downloads, "_plotted", lambda _: frames)
	params = _export_params(csv=True, data={"plot": "network-dominance"})

	resp = test_client.post(
		"/export", data={"figure": json.dumps(fig), "params": json.dumps(params)}
	)

	with zipfile.ZipFile(io.BytesIO(resp.data)) as zf:
		assert set(zf.namelist()) == {"plot.svg", "plot_nodes.csv", "plot_edges.csv"}


def test_export_without_data_is_the_image_alone(client):
	test_client, _ = client
	fig = go.Figure(go.Scatter(x=[1, 2], y=[3, 4])).to_dict()

	resp = test_client.post(
		"/export", data={"figure": json.dumps(fig), "params": json.dumps(_export_params(csv=True))}
	)

	assert resp.mimetype == "image/svg+xml"


def test_export_takes_a_figure_past_werkzeugs_default_form_limit(client):
	# A long recording's timeline posts megabytes of figure JSON, url-encoded on top.
	test_client, _ = client
	points = list(range(100_000))
	fig = go.Figure(go.Scattergl(x=points, y=points)).to_json()
	assert len(fig) > 500_000

	resp = test_client.post(
		"/export", data={"figure": fig, "params": json.dumps(_export_params(format="png"))}
	)

	assert resp.status_code == 200


def test_export_malformed_request_is_a_400(client):
	test_client, _ = client

	resp = test_client.post("/export", data={"figure": "not json"})

	assert resp.status_code == 400
