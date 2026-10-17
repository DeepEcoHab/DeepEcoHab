"""Tests for calculate_chasings (per-hour chasing counts on the dense grid).

calculate_chasings aggregates the event-level match_df into per-(chaser, chased,
tunnel, hour) counts and reindexes them onto the dense ordered-pair x tunnel grid
so absent cells are 0. It reads match_df via Recording.load_results, which we
monkeypatch; the step is called directly.

All events are placed in day-1 light_phase so the run-length phase_count of the
hand-built match_df equals the grid's numbering (the first light phase is 1).
"""

import polars as pl
import strategies

from deepecohab.core import antenna_analysis
from deepecohab.core.data_model import AnalysisParams, Recording

RECORDING = strategies.analysis_recording(animal_ids=["A", "B", "C"])
at = strategies.at


def run_chasings(monkeypatch, match_lf) -> pl.DataFrame:
	monkeypatch.setattr(Recording, "load_results", lambda self, key, eager=False: match_lf)
	return antenna_analysis.calculate_chasings(RECORDING, AnalysisParams()).collect()


def cell(result: pl.DataFrame, chaser: str, chased: str, position: str) -> int:
	"""Total chasings for one ordered pair in one tunnel (summed over the grid)."""
	sub = result.filter(
		(pl.col("chaser") == chaser)
		& (pl.col("chased") == chased)
		& (pl.col("position") == position)
	)
	return int(sub["chasings"].sum())


def test_output_schema_and_zero_fill(monkeypatch):
	"""Result is the dense ordered-pair x tunnel grid with a chasings count."""
	rows = [
		{"winner": "A", "loser": "B", "position": "c1_c2", "datetime": at(2023, 5, 24, 12, 0, 0)}
	]
	result = run_chasings(monkeypatch, strategies.match_df_frame(rows, RECORDING))

	assert {"chaser", "chased", "position", "chasings"}.issubset(set(result.columns))
	# Every ordered pair appears (A!=B etc.), and unobserved cells are 0, not null.
	assert result["chasings"].null_count() == 0
	# Self-pairs are excluded from the ordered-pair grid.
	assert result.filter(pl.col("chaser") == pl.col("chased")).height == 0


def test_counts_per_ordered_pair(monkeypatch):
	"""Two A-over-B chases and one B-over-A chase land in the right directed cells."""
	rows = [
		{"winner": "A", "loser": "B", "position": "c1_c2", "datetime": at(2023, 5, 24, 12, 0, 0)},
		{"winner": "A", "loser": "B", "position": "c1_c2", "datetime": at(2023, 5, 24, 12, 0, 5)},
		{"winner": "B", "loser": "A", "position": "c1_c2", "datetime": at(2023, 5, 24, 12, 0, 9)},
	]
	result = run_chasings(monkeypatch, strategies.match_df_frame(rows, RECORDING))

	assert cell(result, "A", "B", "c1_c2") == 2
	assert cell(result, "B", "A", "c1_c2") == 1


def test_winner_is_chaser_loser_is_chased(monkeypatch):
	"""The match winner becomes the chaser; the count never lands on the reversed cell."""
	rows = [
		{"winner": "A", "loser": "B", "position": "c1_c2", "datetime": at(2023, 5, 24, 12, 0, 0)}
	]
	result = run_chasings(monkeypatch, strategies.match_df_frame(rows, RECORDING))

	assert cell(result, "A", "B", "c1_c2") == 1
	assert cell(result, "B", "A", "c1_c2") == 0


def test_counts_are_per_tunnel(monkeypatch):
	"""A chase in one tunnel does not leak into another tunnel's cell."""
	rows = [
		{"winner": "A", "loser": "B", "position": "c2_c3", "datetime": at(2023, 5, 24, 12, 0, 0)}
	]
	result = run_chasings(monkeypatch, strategies.match_df_frame(rows, RECORDING))

	assert cell(result, "A", "B", "c2_c3") == 1
	assert cell(result, "A", "B", "c1_c2") == 0


def test_empty_match_df_yields_no_rows(monkeypatch):
	"""Only cells with a chasing are stored, so a quiet recording stores none."""
	assert run_chasings(monkeypatch, strategies.match_df_frame([], RECORDING)).is_empty()


def exact_cell(result: pl.DataFrame, chaser: str, chased: str, day: int, hour: int) -> int:
	"""Chasings in one grid cell, rather than summed over the grid as ``cell`` does."""
	return int(
		result.filter(
			(pl.col("chaser") == chaser)
			& (pl.col("chased") == chased)
			& (pl.col("day") == day)
			& (pl.col("hour") == hour)
		)["chasings"].sum()
	)


def test_counts_land_in_the_hour_they_happened_in(monkeypatch):
	"""Aggregation is per cell, so two chases an hour apart do not pool.

	Summing the grid would pass whatever the hour column said, which is how a misplaced
	count hides: the total is right and every time course is wrong.
	"""
	rows = [
		{"winner": "A", "loser": "B", "position": "c1_c2", "datetime": at(2023, 5, 24, 5, 0, 0)},
		{"winner": "A", "loser": "B", "position": "c1_c2", "datetime": at(2023, 5, 25, 5, 30, 0)},
	]
	result = run_chasings(monkeypatch, strategies.match_df_frame(rows, RECORDING))

	assert exact_cell(result, "A", "B", day=1, hour=5) == 1
	assert exact_cell(result, "A", "B", day=2, hour=5) == 1
	assert exact_cell(result, "A", "B", day=1, hour=6) == 0


def test_counts_land_in_the_phase_they_happened_in(monkeypatch):
	"""A chase is numbered by the phase occurrence it happened in.

	The light phase of day 1 is occurrence 1 and the dark phase that follows is 2, so
	two chases either side of the 12:00 switch never share a cell.
	"""
	rows = [
		{"winner": "A", "loser": "B", "position": "c1_c2", "datetime": at(2023, 5, 24, 5, 0, 0)},
		{"winner": "A", "loser": "B", "position": "c1_c2", "datetime": at(2023, 5, 24, 13, 0, 0)},
	]
	result = run_chasings(monkeypatch, strategies.match_df_frame(rows, RECORDING))

	counts = dict(
		result.filter(pl.col("chasings") > 0).select("phase_count", "chasings").iter_rows()
	)
	assert counts == {1: 1, 2: 1}
	assert set(
		result.filter(pl.col("chasings") > 0).select("phase").to_series().cast(pl.String)
	) == {"light_phase", "dark_phase"}
