from types import SimpleNamespace

import polars as pl
import pytest
from pydantic import TypeAdapter, ValidationError

from deepecohab.core.data_model import DataFrameRegistry, FolderName, Project, atomic_write_text


@pytest.mark.parametrize("name", ["..", ".", "a/b", "a\\b", "../x", " "])
def test_folder_name_rejects_paths(name):
	with pytest.raises(ValidationError):
		TypeAdapter(FolderName).validate_python(name)


def test_folder_name_keeps_ordinary_names():
	assert TypeAdapter(FolderName).validate_python("Marcin_2025.11 HET") == "Marcin_2025.11 HET"


def test_failed_step_leaves_no_result_file(tmp_path, monkeypatch):
	def boom(recording, params):
		return pl.LazyFrame({"a": [1]}).select(
			pl.col("a").map_batches(_raise, return_dtype=pl.Int64)
		)

	def _raise(_):
		raise RuntimeError("step died")

	monkeypatch.setitem(DataFrameRegistry._builders, "boom", boom)
	recording = SimpleNamespace(results_path=tmp_path / "results")

	with pytest.raises(Exception, match="step died"):
		DataFrameRegistry.build(recording, "boom", None)  # ty: ignore[invalid-argument-type]

	assert not (recording.results_path / "boom.parquet").exists()


def test_atomic_write_replaces_whole_file(tmp_path):
	path = tmp_path / "x.json"
	atomic_write_text(path, "old")
	atomic_write_text(path, "new")
	assert path.read_text(encoding="utf-8") == "new"
	assert [p.name for p in tmp_path.iterdir()] == ["x.json"]


@pytest.mark.parametrize("name", ["..", "a/b", "a\\b"])
def test_project_create_refuses_a_path_for_a_name_before_making_folders(tmp_path, name):
	with pytest.raises(ValueError, match=r"pattern|folder name"):
		Project.create(name, "tester", tmp_path)

	assert list(tmp_path.iterdir()) == []
