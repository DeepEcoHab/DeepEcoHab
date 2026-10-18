"""Tests for calculate_activity (per-animal occupancy, visits and solitary time).

calculate_activity combines per-position dwell/visit counts (_get_activity) and
solitary time (_get_time_alone) from padded_df, then reindexes onto the dense
animal x position x hour grid (reindex_onto_grid), zero-filling empty cells.
padded_df is read via Recording.load_results (monkeypatched); the body runs via
directly. Events sit in day-1 light_phase so phase_count matches the grid.
"""

import datetime as dt

import polars as pl
import strategies

from deepecohab.core import antenna_analysis
from deepecohab.core.data_model import AnalysisParams, Recording

RECORDING = strategies.analysis_recording(animal_ids=["A", "B", "C"])
at = strategies.at


def run_activity(monkeypatch, padded_lf, minimum_time_alone: float = 0.0) -> pl.DataFrame:
	"""These cases exercise the sweep, so the solitary-span threshold is off by default."""
	monkeypatch.setattr(Recording, "load_results", lambda self, key, eager=False: padded_lf)
	params = AnalysisParams(minimum_time_alone=minimum_time_alone)
	return antenna_analysis.calculate_activity(RECORDING, params).collect()


def cell(result: pl.DataFrame, animal: str, position: str) -> dict:
	"""Sum the metrics for one animal/position over the (single) active hour."""
	sub = result.filter((pl.col("animal_id") == animal) & (pl.col("position") == position))
	return {
		"time_in_position": sub["time_in_position"].sum(),
		"visits_to_position": sub["visits_to_position"].sum(),
		"time_alone": sub["time_alone"].sum(),
	}


def test_output_schema_and_dense_grid(monkeypatch):
	"""Result spans every animal x position cell with the three metric columns."""
	rows = [
		{
			"animal_id": "A",
			"position": "cage_1",
			"datetime": at(2023, 5, 24, 12, 0, 5),
			"time_spent": 5.0,
		}
	]
	result = run_activity(monkeypatch, strategies.padded_df_frame(rows, RECORDING))

	assert {"time_in_position", "visits_to_position", "time_alone"}.issubset(set(result.columns))
	# Every animal and every position appears (dense grid), with no nulls.
	assert set(result["animal_id"].unique()) == set(RECORDING.cohort.animal_tags)
	assert set(result["position"].cast(pl.String).unique()) == set(
		RECORDING.layout.positions_non_directional
	)
	assert (
		result.select(pl.col("time_in_position", "visits_to_position", "time_alone"))
		.null_count()
		.sum_horizontal()
		.item()
		== 0
	)


def test_interpolated_reads_excluded_from_visits_but_keep_time(monkeypatch):
	"""An interpolated piece adds its dwell time but is not counted as a visit."""
	rows = [
		{
			"animal_id": "A",
			"position": "cage_1",
			"datetime": at(2023, 5, 24, 12, 0, 5),
			"time_spent": 5.0,
			"interpolated": False,
		},
		{
			"animal_id": "A",
			"position": "cage_1",
			"datetime": at(2023, 5, 24, 12, 0, 10),
			"time_spent": 3.0,
			"interpolated": True,
		},
	]
	c = cell(run_activity(monkeypatch, strategies.padded_df_frame(rows, RECORDING)), "A", "cage_1")

	assert c["time_in_position"] == dt.timedelta(seconds=8.0)  # both pieces contribute their time
	assert c["visits_to_position"] == 1  # only the non-interpolated read is a visit


def test_solitary_time_recorded(monkeypatch):
	"""A lone animal's time_alone equals its occupancy for that cell."""
	rows = [
		{
			"animal_id": "A",
			"position": "cage_1",
			"datetime": at(2023, 5, 24, 12, 0, 10),
			"time_spent": 10.0,
		}
	]
	c = cell(run_activity(monkeypatch, strategies.padded_df_frame(rows, RECORDING)), "A", "cage_1")

	assert c["time_in_position"] == dt.timedelta(seconds=10.0)
	assert c["time_alone"] == dt.timedelta(seconds=10.0)


def test_minimum_time_alone_drops_only_shorter_spans(monkeypatch):
	"""Brief solitary spans - the gap left by a group arriving moments apart - are cut.

	The threshold is strict, matching minimum_time for meetings: a span exactly at it
	does not survive.
	"""
	rows = [
		{
			"animal_id": "A",
			"position": "cage_1",
			"datetime": at(2023, 5, 24, 12, 0, 4),
			"time_spent": 4.0,
		},
		{
			"animal_id": "B",
			"position": "cage_2",
			"datetime": at(2023, 5, 24, 12, 0, 30),
			"time_spent": 30.0,
		},
	]
	padded = strategies.padded_df_frame(rows, RECORDING)

	kept = run_activity(monkeypatch, padded, minimum_time_alone=0.0)
	assert cell(kept, "A", "cage_1")["time_alone"] == dt.timedelta(seconds=4.0)
	assert cell(kept, "B", "cage_2")["time_alone"] == dt.timedelta(seconds=30.0)

	filtered = run_activity(monkeypatch, padded, minimum_time_alone=10.0)
	assert cell(filtered, "A", "cage_1")["time_alone"] == dt.timedelta(0)
	assert cell(filtered, "B", "cage_2")["time_alone"] == dt.timedelta(seconds=30.0)
	# Occupancy is untouched: only the solitary reading is filtered.
	assert cell(filtered, "A", "cage_1")["time_in_position"] == dt.timedelta(seconds=4.0)

	at_threshold = run_activity(monkeypatch, padded, minimum_time_alone=30.0)
	assert cell(at_threshold, "B", "cage_2")["time_alone"] == dt.timedelta(0)


def test_directional_tunnel_ends_collapse_into_one_position(monkeypatch):
	"""The two ends of a tunnel are one place, so their dwell time is one cell.

	padded_df carries directional positions (``c1_c2``/``c2_c1``); without collapsing
	them a tunnel's occupancy would be split across two names, neither of which the
	dense grid holds - and the reindex would drop both.
	"""
	rows = [
		{
			"animal_id": "A",
			"position": "c1_c2",
			"datetime": at(2023, 5, 24, 12, 0, 5),
			"time_spent": 5.0,
		},
		{
			"animal_id": "A",
			"position": "c2_c1",
			"datetime": at(2023, 5, 24, 12, 0, 12),
			"time_spent": 2.0,
		},
	]
	result = run_activity(monkeypatch, strategies.padded_df_frame(rows, RECORDING))

	assert cell(result, "A", "tunnel_1")["time_in_position"] == dt.timedelta(seconds=7.0)
	assert cell(result, "A", "tunnel_1")["visits_to_position"] == 2
	# Neither directional name survives into the table.
	assert "c1_c2" not in set(result["position"].cast(pl.String).unique())


def test_opposite_directions_share_the_tunnel_so_neither_is_alone(monkeypatch):
	"""Collapsing directionality is what lets two animals meet in one tunnel.

	They pass through the same tunnel in opposite directions at the same time. Each one
	holds its own occupancy, but the overlap belongs to neither as solitary time.
	"""
	rows = [
		{
			"animal_id": "A",
			"position": "c1_c2",
			"datetime": at(2023, 5, 24, 12, 0, 10),
			"time_spent": 10.0,
		},
		{
			"animal_id": "B",
			"position": "c2_c1",
			"datetime": at(2023, 5, 24, 12, 0, 10),
			"time_spent": 10.0,
		},
	]
	result = run_activity(monkeypatch, strategies.padded_df_frame(rows, RECORDING))

	for animal in ("A", "B"):
		assert cell(result, animal, "tunnel_1")["time_in_position"] == dt.timedelta(seconds=10.0)
		assert cell(result, animal, "tunnel_1")["time_alone"] == dt.timedelta(0)


def test_undefined_carries_time_but_never_solitary_time(monkeypatch):
	"""``undefined`` is time the analysis has to account for but cannot place.

	It keeps its occupancy - the tiling of each animal's timeline depends on it - while
	contributing no solitary time, because being alone somewhere unknown is not a fact
	about any position.
	"""
	rows = [
		{
			"animal_id": "A",
			"position": "undefined",
			"datetime": at(2023, 5, 24, 12, 0, 30),
			"time_spent": 30.0,
		}
	]
	result = run_activity(monkeypatch, strategies.padded_df_frame(rows, RECORDING))

	assert cell(result, "A", "undefined")["time_in_position"] == dt.timedelta(seconds=30.0)
	assert cell(result, "A", "undefined")["visits_to_position"] == 1
	assert cell(result, "A", "undefined")["time_alone"] == dt.timedelta(0)


def test_empty_padded_df_still_yields_the_dense_grid(monkeypatch):
	"""A recording where nothing was registered at all produces zeros, not nothing.

	Every downstream step left-joins onto this table, so an empty result rather than an
	all-zero grid would propagate as missing rows instead of as observed silence.
	"""
	result = run_activity(monkeypatch, strategies.padded_df_frame([], RECORDING))

	assert result.height > 0
	assert set(result["animal_id"].unique()) == set(RECORDING.cohort.animal_tags)
	assert result["time_in_position"].sum() == dt.timedelta(0)
	assert result["visits_to_position"].sum() == 0
	assert result["time_alone"].sum() == dt.timedelta(0)
