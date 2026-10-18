"""Run analysis starts in the server's own process and reports progress as a snapshot."""

import time
from pathlib import Path

import pytest
from dash import Dash
from test_project_table import make_recording, write_recording

from deepecohab import Project
from deepecohab.app import services

Dash(__name__, use_pages=True, pages_folder="")


@pytest.fixture
def selection(tmp_path, monkeypatch):
	monkeypatch.setattr(services, "notify", lambda *args: None)
	project = Project.create(project_name="p", experimenter="t", location=tmp_path / "project")
	project.add_recordings(
		path
		for name in ("r1", "r2")
		for path in write_recording(
			tmp_path / "src", make_recording(name, ["A", "B"], "WT", "2023-05-26 00:00:00"), 12
		)
	)
	location = str(project.project_location)
	return [[location, "r1"], [location, "r2"]]


def _finish(run_id):
	deadline = time.monotonic() + 120
	while (finished := services.finish_run(run_id, {})) is None:
		assert time.monotonic() < deadline
		time.sleep(0.05)
	return finished


def test_a_run_reports_every_selected_recording_and_builds_them(selection):
	location = selection[0][0]
	run_id = services.start_run(selection, None, overwrite=False, use_prev_ranking=True)

	assert set(services.run_progress(run_id)[location]) == {"r1", "r2"}
	failed, cleared = _finish(run_id)

	assert failed == {location: {}}
	assert cleared == []
	assert all(services.recording_status(Path(location) / name).values() for _, name in selection)
	assert not services.analysis_running()


def test_nothing_to_build_starts_no_run(selection):
	run_id = services.start_run(selection, None, overwrite=False, use_prev_ranking=True)
	_finish(run_id)
	assert services.start_run(selection, None, overwrite=False, use_prev_ranking=True) is None
