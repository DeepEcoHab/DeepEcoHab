from pathlib import Path

import pytest

from deepecohab.core.data_model import _group_files

CASE_SENSITIVE_PATHS = Path("a") != Path("A")


def names(groups):
	return {base.name: {s: p.name for s, p in files.items()} for base, files in groups.items()}


def errors(failed):
	return {f.name: type(f.error) for f in failed}


def reasons(failed):
	return {f.name: str(f.error) for f in failed}


def recording(folder, name="a"):
	"""Paths of one complete recording's files."""
	return [folder / f"{name}.{s}" for s in ("config.json", "data.parquet", "diagnostic.json")]


def complete(name="a"):
	return {s: f"{name}.{s}" for s in ("config.json", "data.parquet", "diagnostic.json")}


FORMAT = (
	"does not match the file format "
	"<name>.config.json, <name>.data.parquet or <name>.diagnostic.json"
)


def test_pairs_files_by_name(tmp_path):
	groups, failed = _group_files(
		[
			tmp_path / "a.config.json",
			tmp_path / "b.data.parquet",
			tmp_path / "a.data.parquet",
			tmp_path / "a.diagnostic.json",
			tmp_path / "b.config.json",
			tmp_path / "b.diagnostic.json",
		]
	)
	assert failed == []
	assert names(groups) == {"a": complete("a"), "b": complete("b")}


@pytest.mark.parametrize(
	("present", "reason"),
	[
		(
			["a.config.json"],
			(
				"no data file or diagnostic file - expected a.data.parquet and a.diagnostic.json "
				"beside a.config.json"
			),
		),
		(
			["a.config.json", "a.data.parquet"],
			(
				"no diagnostic file - expected a.diagnostic.json beside a.config.json and "
				"a.data.parquet"
			),
		),
		(
			["a.data.parquet", "a.diagnostic.json"],
			"no config file - expected a.config.json beside a.data.parquet and a.diagnostic.json",
		),
		(
			["a.diagnostic.json"],
			(
				"no config file or data file - expected a.config.json and a.data.parquet "
				"beside a.diagnostic.json"
			),
		),
	],
)
def test_a_set_without_a_required_file_fails_naming_what_is_missing(tmp_path, present, reason):
	groups, failed = _group_files([tmp_path / name for name in present])
	assert groups == {}
	assert errors(failed) == {"a": FileNotFoundError}
	assert reasons(failed) == {"a": reason}


def test_an_incomplete_set_does_not_block_the_rest(tmp_path):
	groups, failed = _group_files([tmp_path / "a.config.json", *recording(tmp_path, "b")])
	assert list(names(groups)) == ["b"]
	assert errors(failed) == {"a": FileNotFoundError}


@pytest.mark.parametrize(
	"stray",
	["notes.txt", "a.json", "a.parquet", "a.config.json.bak", "a.config.json.json", "config"],
)
def test_a_stray_file_fails_on_its_own_naming_the_format(tmp_path, stray):
	groups, failed = _group_files([tmp_path / stray, *recording(tmp_path)])
	assert list(names(groups)) == ["a"]
	assert errors(failed) == {stray: ValueError}
	assert reasons(failed) == {stray: FORMAT}


@pytest.mark.parametrize(
	"nameless",
	[".config.json", ". .config.json"],
)
def test_a_file_with_no_name_before_its_suffix_fails_without_raising(tmp_path, nameless):
	groups, failed = _group_files([tmp_path / nameless, tmp_path / "..data.parquet"])
	assert groups == {}
	assert errors(failed) == {nameless: ValueError, "..data.parquet": ValueError}
	assert reasons(failed) == {
		nameless: "has no recording name before .config.json",
		"..data.parquet": "has no recording name before .data.parquet",
	}


def test_a_name_may_contain_dots_and_other_suffixes(tmp_path):
	groups, failed = _group_files(recording(tmp_path, "rec.data.parquet"))
	assert failed == []
	assert list(names(groups)) == ["rec.data.parquet"]


def test_suffix_case_is_ignored(tmp_path):
	groups, failed = _group_files(
		[tmp_path / "a.CONFIG.JSON", tmp_path / "a.Data.Parquet", tmp_path / "a.DIAGNOSTIC.json"]
	)
	assert failed == []
	assert names(groups) == {
		"a": {
			"config.json": "a.CONFIG.JSON",
			"data.parquet": "a.Data.Parquet",
			"diagnostic.json": "a.DIAGNOSTIC.json",
		}
	}


def test_name_case_is_ignored_and_the_first_spelling_kept(tmp_path):
	groups, failed = _group_files([tmp_path / "Rec1.config.json", *recording(tmp_path, "rec1")[1:]])
	assert failed == []
	assert list(names(groups)) == ["Rec1"]


def test_same_name_in_different_folders_is_different_recordings(tmp_path):
	groups, failed = _group_files([*recording(tmp_path / "x"), *recording(tmp_path / "y")])
	assert failed == []
	assert sorted(groups) == [tmp_path.resolve() / "x" / "a", tmp_path.resolve() / "y" / "a"]


def test_different_spellings_of_one_folder_are_one_folder(tmp_path, monkeypatch):
	monkeypatch.chdir(tmp_path)
	groups, failed = _group_files(
		["x/a.config.json", tmp_path / "x" / ".." / "x" / "a.data.parquet", "x/a.diagnostic.json"]
	)
	assert failed == []
	assert list(groups) == [tmp_path.resolve() / "x" / "a"]


def test_the_same_file_given_twice_counts_once(tmp_path):
	config = tmp_path / "a.config.json"
	groups, failed = _group_files([*recording(tmp_path), config, str(config)])
	assert failed == []
	assert names(groups) == {"a": complete()}


@pytest.mark.skipif(not CASE_SENSITIVE_PATHS, reason="one file on a case-insensitive OS")
@pytest.mark.parametrize("twin", ["A.config.json", "a.CONFIG.json"])
def test_two_files_for_one_suffix_fail_the_set(tmp_path, twin):
	groups, failed = _group_files([*recording(tmp_path), tmp_path / twin])
	assert groups == {}
	assert errors(failed) == {"a": ValueError}
	assert reasons(failed) == {"a": f"2 config files, a.config.json and {twin} - keep one"}


@pytest.mark.skipif(CASE_SENSITIVE_PATHS, reason="two files on a case-sensitive OS")
def test_case_variants_of_one_path_are_one_file(tmp_path):
	groups, failed = _group_files([*recording(tmp_path), tmp_path / "A.CONFIG.json"])
	assert failed == []
	assert names(groups) == {"a": complete()}
