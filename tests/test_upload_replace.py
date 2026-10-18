"""Re-uploading recordings whose names are taken, through the Projects page dialogs."""

import base64
from pathlib import Path

import pytest
from dash import Dash
from dash._callback_context import context_value
from dash._utils import AttributeDict
from test_project_table import make_recording, write_recording

from deepecohab import Project
from deepecohab.app import services

Dash(__name__, use_pages=True, pages_folder="")

from deepecohab.app.pages import projects  # noqa: E402

_START = "2023-05-26 00:00:00"
toasts: list[tuple[str, str]] = []


def _fire(callback, prop_id, *args):
	context_value.set(AttributeDict(triggered_inputs=[{"prop_id": prop_id, "value": 1}]))
	try:
		return callback(*args)
	finally:
		context_value.set({})


def _upload(project: Project, names: list[str], genotype: str, tmp_path: Path) -> dict:
	"""Upload recordings called ``names`` with ``genotype``; returns the pending store."""
	filenames, contents = [], []
	for name in names:
		recording = make_recording(name, ["A", "B"], genotype, _START)
		for path in write_recording(tmp_path / genotype, recording, 12):
			filenames.append(path.name)
			contents.append("data:;base64," + base64.b64encode(path.read_bytes()).decode())
	outputs = _fire(
		projects._add_recordings,
		"upload-files.contents",
		None,
		contents,
		None,
		filenames,
		str(project.project_location),
	)
	return outputs[-1]


@pytest.fixture
def project(tmp_path, monkeypatch):
	monkeypatch.setattr(projects, "notify", lambda kind, message: toasts.append((kind, message)))
	monkeypatch.setattr(services, "load_project", lambda location: Project.load(Path(location)))
	project = Project.create(project_name="up", experimenter="t", location=tmp_path / "project")
	project.add_recordings(
		path
		for name in ("r1", "r2", "r3")
		for path in write_recording(
			tmp_path / "src", make_recording(name, ["A", "B"], "WT", _START), 12
		)
	)
	return project


def _genotypes(project: Project) -> dict[str, str]:
	reloaded = Project.load(project.project_location)
	return {r.name: r.cohort.animals[0].genotype for r in reloaded.recordings}


def test_nothing_is_replaced_until_confirmed_and_apply_to_all_covers_the_rest(project, tmp_path):
	with pytest.warns(UserWarning, match="left as they are"):
		pending = _upload(project, ["r1", "r2", "r3"], "KO", tmp_path)
	assert sorted(pending["existing"]) == ["r1", "r2", "r3"]
	assert set(_genotypes(project).values()) == {"WT"}

	opened, _, label, *_, queue, _ = _fire(
		projects._replace_recordings, "replace-pending.data", pending, None, None, None, False
	)
	assert opened
	assert "2 other recordings" in label

	# Skip the first, then replace the remaining two at once.
	opened, *_, queue, _ = _fire(
		projects._replace_recordings, "replace-skip.n_clicks", pending, 1, None, queue, False
	)
	assert opened
	assert queue["waiting"] == ["r2", "r3"]
	with pytest.warns(UserWarning, match="Replaced 2 recordings"):
		opened, *_ = _fire(
			projects._replace_recordings, "replace-confirm.n_clicks", pending, 1, 1, queue, True
		)

	assert not opened
	assert _genotypes(project) == {"r1": "WT", "r2": "KO", "r3": "KO"}
	assert toasts[-1][0] == "warn"
	assert "Replaced r2, r3" in toasts[-1][1]
	assert not Path(pending["staging"]).exists()
