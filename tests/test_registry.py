"""Tests for DataFrameRegistry dependency resolution (step_order).

The topological sort is the one piece of pipeline orchestration that is pure and
free of I/O, so it is unit-testable in isolation. Steps register on the class, so
these tests swap in a throwaway graph, assert ordering invariants, and then check
the real registered pipeline.
"""

import polars as pl
import pytest
import strategies

from deepecohab.core.data_model import DataFrameRegistry, recording_status


@pytest.fixture
def graph(monkeypatch):
	"""Install a throwaway dependency graph in place of the registered pipeline."""

	def _install(deps: dict[str, list[str]]) -> type[DataFrameRegistry]:
		monkeypatch.setattr(
			DataFrameRegistry, "_builders", dict.fromkeys(deps, lambda recording, params: None)
		)
		monkeypatch.setattr(
			DataFrameRegistry, "_requires", {name: list(r) for name, r in deps.items()}
		)
		return DataFrameRegistry

	return _install


def _is_topo(order: list[str], deps: dict[str, list[str]]) -> bool:
	"""Every dependency that is itself a step must appear before its dependant."""
	pos = {name: i for i, name in enumerate(order)}
	steps = set(deps)
	return all(
		pos[req] < pos[name] for name, requires in deps.items() for req in requires if req in steps
	)


def test_linear_chain_orders_dependencies_first(graph):
	deps = {"a": [], "b": ["a"], "c": ["b"]}
	assert graph(deps).step_order() == ["a", "b", "c"]


def test_unregistered_requirement_raises(graph):
	"""Every requirement must be produced by some step, or the graph is incomplete."""
	deps = {"x": ["main_df"], "y": ["main_df"]}
	with pytest.raises(KeyError, match="no step produces"):
		graph(deps).step_order()


def test_diamond_is_valid_topological_order(graph):
	deps = {"root": [], "left": ["root"], "right": ["root"], "join": ["left", "right"]}
	order = graph(deps).step_order()
	assert _is_topo(order, deps)
	assert order[0] == "root"
	assert order[-1] == "join"


def test_order_is_deterministic(graph):
	deps = {"a": [], "b": [], "c": ["a", "b"]}
	registry = graph(deps)
	assert registry.step_order() == registry.step_order()


def test_targets_returns_only_transitive_closure(graph):
	deps = {"a": [], "b": ["a"], "c": ["b"], "other": []}
	order = graph(deps).step_order(targets=["c"])
	assert order == ["a", "b", "c"]
	assert "other" not in order


def test_cycle_raises(graph):
	deps = {"a": ["b"], "b": ["a"]}
	with pytest.raises(ValueError, match="Cycle"):
		graph(deps).step_order()


def test_unknown_target_raises(graph):
	with pytest.raises(KeyError, match="Unknown analysis step"):
		graph({"a": []}).step_order(targets=["nope"])


def test_real_pipeline_order_respects_dependencies():
	order = DataFrameRegistry.step_order()
	# The data structure comes first, then match_df before its consumers, and every
	# feature input before feature_df.
	assert order.index("main_df") < order.index("padded_df")
	assert order.index("padded_df") < order.index("activity_df")
	assert order.index("match_df") < order.index("chasings_df")
	assert order.index("match_df") < order.index("ranking")
	for dependency in ("chasings_df", "pairwise_meetings", "activity_df"):
		assert order.index(dependency) < order.index("feature_df")


def test_auxiliary_analysis_registers_nothing():
	"""Importing an auxiliary analysis must not add a step to the pipeline."""
	from deepecohab.auxiliary_analysis import tube_test  # noqa: F401  import is the point

	assert "tube_test_df" not in DataFrameRegistry.list_available()


def test_load_results_reads_a_table_no_step_produces(tmp_path):
	"""An auxiliary table sunk into results/ loads back, though nothing registers it."""
	recording = strategies.analysis_recording(animal_ids=["A", "B"])
	recording._root = tmp_path
	(tmp_path / "results").mkdir()
	pl.DataFrame({"winner": ["A"], "loser": ["B"]}).write_parquet(
		tmp_path / "results" / "tube_test_df.parquet"
	)

	assert recording.load_results("tube_test_df", eager=True).height == 1

	# A name with neither a step nor a parquet is still a typo, and still reported as one.
	with pytest.raises(KeyError, match="not a registered data key"):
		recording.load_results("tube_test_de")


def test_recording_status_reads_results_without_a_recording(tmp_path):
	"""A project whose config.json fails to validate must still report status."""
	(tmp_path / "results").mkdir()
	(tmp_path / "results" / "main_df.parquet").touch()

	status = recording_status(tmp_path)

	assert status["main_df"] is True
	assert status["padded_df"] is False
	assert set(status) == set(DataFrameRegistry.step_order())


def test_recording_status_of_an_unanalysed_recording_is_all_false(tmp_path):
	assert not any(recording_status(tmp_path).values())
