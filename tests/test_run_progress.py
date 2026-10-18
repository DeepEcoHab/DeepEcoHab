"""Clicking Run analysis paints step 0 before the background run first reports."""

from pathlib import Path

import pytest
from dash import Dash
from dash._callback_context import context_value
from dash._utils import AttributeDict
from test_project_table import make_recording, write_recording

from deepecohab import Project
from deepecohab.app import services
from deepecohab.core.data_model import DataFrameRegistry

Dash(__name__, use_pages=True, pages_folder="")

from deepecohab.app.pages import projects  # noqa: E402


def _actionbar(*args):
	context = AttributeDict(
		triggered_inputs=[{"prop_id": "run-active.data", "value": True}], updated_props={}
	)
	context_value.set(context)
	try:
		return projects._render_actionbar(*args), context.updated_props
	finally:
		context_value.set({})


@pytest.fixture
def selection(tmp_path, monkeypatch):
	monkeypatch.setattr(services, "load_project", lambda location: Project.load(Path(location)))
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


def test_run_start_seeds_progress_for_the_recordings_it_will_build(selection):
	order = DataFrameRegistry.step_order()
	(hidden, classes, message, _), updated = _actionbar(selection, None, True, False)

	location = selection[0][0]
	seeded = {location: {"r1": [0, len(order), order[0]], "r2": [0, len(order), order[0]]}}
	assert updated["run-progress"] == {"data": seeded}
	assert not hidden
	assert "is-running" in classes
	assert f"0/{2 * len(order)} steps" in str(message)


def test_nothing_to_build_seeds_empty_progress_once(selection, monkeypatch):
	monkeypatch.setattr(services, "_analysed", lambda location, name: True)
	(_, classes, _, _), updated = _actionbar(selection, None, True, False)
	assert updated["run-progress"] == {"data": {}}
	assert "is-running" not in classes

	# The seeded {} comes back as an input; it must not seed again.
	_, updated = _actionbar(selection, {}, True, False)
	assert updated == {}
