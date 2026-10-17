"""Tests for calculate_pairwise_meetings (co-occurrence time and encounter counts).

Each animal's back-to-back registrations in a position (a cage or a tunnel, with tunnel
directionality collapsed first, as _get_time_alone sees them via _occupancy_intervals)
merge into a stay; two animals' overlapping stays are one meeting, meetings shorter
than ``minimum_time`` are dropped and the rest summed onto the dense grid. main_df is
read via Recording.load_results (monkeypatched); the step is called directly.

The rows carry the interval END in ``datetime`` and its length in ``time_spent``; all
events sit in day-1 light_phase.
"""

import datetime as dt

import polars as pl
import strategies

from deepecohab.core import antenna_analysis, transforms
from deepecohab.core.data_model import AnalysisParams, Recording

RECORDING = strategies.analysis_recording(animal_ids=["A", "B", "C"])
at = strategies.at


def padded_pieces(rows, recording) -> pl.LazyFrame:
	"""Split each ``stay`` row at cell starts with the real padding, as padded_df does.

	Feeds the actual splitter so a stay crossing cells becomes several pieces
	(interpolated flags and all). The pairwise step reads them as registrations, so
	this exercises the split -> merge-back path.
	"""
	df = pl.DataFrame(
		{
			"animal_id": pl.Series(
				[r["animal_id"] for r in rows], dtype=pl.Enum(recording.cohort.animal_tags)
			),
			"position": pl.Series([r["position"] for r in rows], dtype=pl.Categorical),
			"datetime": pl.Series("datetime", [r["datetime"] for r in rows]),
			"time_spent": strategies.seconds([float(r["time_spent"]) for r in rows]),
			"time_under": pl.Series([dt.timedelta(0) for _ in rows], dtype=pl.Duration("us")),
		}
	)
	return transforms.split_visits_on_cells(df.lazy(), recording)


def run_pairwise(monkeypatch, padded_lf, **kwargs) -> pl.DataFrame:
	monkeypatch.setattr(Recording, "load_results", lambda self, key, eager=False: padded_lf)
	return antenna_analysis.calculate_pairwise_meetings(
		RECORDING, AnalysisParams(**kwargs)
	).collect()


def pair_cell(result: pl.DataFrame, a: str, b: str, position: str) -> dict:
	"""Co-presence totals for the unordered pair (a, b) in one cage."""
	lo, hi = sorted([a, b])
	sub = result.filter(
		(pl.col("animal_id") == lo)
		& (pl.col("animal_id_2") == hi)
		& (pl.col("position") == position)
	)
	return {
		"time_together": sub["time_together"].sum(),
		"pairwise_encounters": sub["pairwise_encounters"].sum(),
	}


def stay(animal, position, end, length):
	"""One padded_df occupancy row: animal in position ending at `end` for `length` s."""
	return {"animal_id": animal, "position": position, "datetime": end, "time_spent": float(length)}


def test_output_schema_and_unordered_pairs(monkeypatch):
	"""Result carries time_together/pairwise_encounters; pairs are unordered (a < b)."""
	rows = [
		stay("A", "cage_1", at(2023, 5, 24, 12, 0, 10), 10),
		stay("B", "cage_1", at(2023, 5, 24, 12, 0, 15), 10),
	]
	result = run_pairwise(monkeypatch, strategies.padded_df_frame(rows, RECORDING))

	assert {"time_together", "pairwise_encounters"}.issubset(set(result.columns))
	# unordered: animal_id is always the lexicographically smaller of the pair.
	assert (result["animal_id"].cast(pl.String) < result["animal_id_2"].cast(pl.String)).all()
	assert (
		result.select(pl.col("time_together", "pairwise_encounters"))
		.null_count()
		.sum_horizontal()
		.item()
		== 0
	)


def test_shared_time_and_one_encounter(monkeypatch):
	"""Overlapping cage stays yield exactly the overlap and a single encounter."""
	# A in cage_1 [12:00:00, 12:00:10]; B [12:00:05, 12:00:15]; overlap 5 s.
	rows = [
		stay("A", "cage_1", at(2023, 5, 24, 12, 0, 10), 10),
		stay("B", "cage_1", at(2023, 5, 24, 12, 0, 15), 10),
	]
	c = pair_cell(
		run_pairwise(monkeypatch, strategies.padded_df_frame(rows, RECORDING)), "A", "B", "cage_1"
	)
	assert c["time_together"] == dt.timedelta(seconds=5.0)
	assert c["pairwise_encounters"] == 1


def test_short_meeting_dropped_by_minimum_time(monkeypatch):
	"""A meeting shorter than minimum_time contributes nothing (cell stays zero)."""
	# overlap of 1 s, below the default minimum_time of 2 s.
	rows = [
		stay("A", "cage_1", at(2023, 5, 24, 12, 0, 10), 10),
		stay("B", "cage_1", at(2023, 5, 24, 12, 0, 19), 10),
	]
	c = pair_cell(
		run_pairwise(monkeypatch, strategies.padded_df_frame(rows, RECORDING)), "A", "B", "cage_1"
	)
	assert c["time_together"] == dt.timedelta(seconds=0.0)
	assert c["pairwise_encounters"] == 0


def test_minimum_time_zero_keeps_short_meeting(monkeypatch):
	"""With minimum_time=0 the same 1 s meeting is retained."""
	rows = [
		stay("A", "cage_1", at(2023, 5, 24, 12, 0, 10), 10),
		stay("B", "cage_1", at(2023, 5, 24, 12, 0, 19), 10),
	]
	c = pair_cell(
		run_pairwise(monkeypatch, strategies.padded_df_frame(rows, RECORDING), minimum_time=0),
		"A",
		"B",
		"cage_1",
	)
	assert c["time_together"] == dt.timedelta(seconds=1.0)
	assert c["pairwise_encounters"] == 1


def test_three_animals_decode_all_pairs(monkeypatch):
	"""Three animals sharing a cage produce all three pairs (bitmask decode)."""
	# Identical interval [12:00:00, 12:00:10] for A, B and C.
	rows = [
		stay("A", "cage_1", at(2023, 5, 24, 12, 0, 10), 10),
		stay("B", "cage_1", at(2023, 5, 24, 12, 0, 10), 10),
		stay("C", "cage_1", at(2023, 5, 24, 12, 0, 10), 10),
	]
	result = run_pairwise(monkeypatch, strategies.padded_df_frame(rows, RECORDING))
	for a, b in (("A", "B"), ("A", "C"), ("B", "C")):
		c = pair_cell(result, a, b, "cage_1")
		assert c["time_together"] == dt.timedelta(seconds=10.0), f"pair {a},{b}"
		assert c["pairwise_encounters"] == 1, f"pair {a},{b}"


def test_separate_meetings_counted_twice(monkeypatch):
	"""Two disjoint co-presence bouts of the same pair count as two encounters."""
	rows = [
		stay("A", "cage_1", at(2023, 5, 24, 12, 0, 5), 5),
		stay("B", "cage_1", at(2023, 5, 24, 12, 0, 5), 5),
		stay("A", "cage_1", at(2023, 5, 24, 12, 0, 25), 5),
		stay("B", "cage_1", at(2023, 5, 24, 12, 0, 25), 5),
	]
	c = pair_cell(
		run_pairwise(monkeypatch, strategies.padded_df_frame(rows, RECORDING)), "A", "B", "cage_1"
	)
	assert c["pairwise_encounters"] == 2
	assert c["time_together"] == dt.timedelta(seconds=10.0)


def test_contiguous_spans_stitched_into_one_meeting(monkeypatch):
	"""A continuous A/B meeting split by a third animal's events is one encounter.

	C enters and leaves while A and B stay put, so the sweep cuts the A/B span into
	contiguous pieces; stitching must recombine them into a single meeting.
	"""
	rows = [
		stay("A", "cage_1", at(2023, 5, 24, 12, 0, 20), 20),  # [12:00:00, 12:00:20]
		stay("B", "cage_1", at(2023, 5, 24, 12, 0, 20), 20),  # [12:00:00, 12:00:20]
		stay("C", "cage_1", at(2023, 5, 24, 12, 0, 10), 5),  # [12:00:05, 12:00:10]
	]
	c = pair_cell(
		run_pairwise(monkeypatch, strategies.padded_df_frame(rows, RECORDING)), "A", "B", "cage_1"
	)
	assert c["pairwise_encounters"] == 1
	assert c["time_together"] == dt.timedelta(seconds=20.0)


def test_continuous_costay_split_across_cells_is_one_encounter(monkeypatch):
	"""A single continuous co-stay counts once, even though padding splits it.

	Regression for the elevated pairwise_encounters: A and B sit together in a cage
	across two cell starts of the grid. Padding cuts that stay into per-cell pieces,
	but reconstructing exact interval bounds lets them merge back into ONE meeting. A
	boundary-rounding bug fragments the pieces into many spurious encounters instead.
	The fractional-second offset makes it sensitive to that bug.
	"""
	# both continuously in cage_1 over [11:58:20.5, 13:01:40.5]: across 12:00 and 13:00.
	start = at(2023, 5, 24, 11, 58, 20, 500_000)
	end = at(2023, 5, 24, 13, 1, 40, 500_000)
	length = (end - start).total_seconds()
	rows = [
		{"animal_id": "A", "position": "cage_1", "datetime": end, "time_spent": length},
		{"animal_id": "B", "position": "cage_1", "datetime": end, "time_spent": length},
	]
	padded = padded_pieces(rows, RECORDING)
	# sanity: the stay really was split into multiple pieces per animal.
	assert padded.collect().filter(pl.col("animal_id") == "A").height > 1

	c = pair_cell(run_pairwise(monkeypatch, padded), "A", "B", "cage_1")
	assert c["pairwise_encounters"] == 1
	assert c["time_together"] == dt.timedelta(seconds=length)


def test_encounters_never_exceed_sum_of_visits(monkeypatch):
	"""User's invariant: a pair can't meet more often than their combined visit count.

	A has three cage_1 visits and B two; some overlap, some don't. However the
	meetings fall out, the encounter count for the pair in that cage cannot exceed
	3 + 2 = 5 -- a pair meeting requires at least one of them to (re-)enter.
	"""

	def visit(a, s, e):
		return {
			"animal_id": a,
			"position": "cage_1",
			"datetime": e,
			"time_spent": (e - s).total_seconds(),
		}

	rows = [
		visit("A", at(2023, 5, 24, 12, 0, 0), at(2023, 5, 24, 12, 0, 10)),
		visit("B", at(2023, 5, 24, 12, 0, 5), at(2023, 5, 24, 12, 0, 15)),  # overlaps A#1
		visit("A", at(2023, 5, 24, 12, 0, 30), at(2023, 5, 24, 12, 0, 40)),  # alone
		visit("A", at(2023, 5, 24, 12, 1, 0), at(2023, 5, 24, 12, 1, 20)),
		visit("B", at(2023, 5, 24, 12, 1, 10), at(2023, 5, 24, 12, 1, 25)),  # overlaps A#3
	]
	result = run_pairwise(monkeypatch, strategies.padded_df_frame(rows, RECORDING), minimum_time=0)
	c = pair_cell(result, "A", "B", "cage_1")
	visits_a, visits_b = 3, 2
	assert c["pairwise_encounters"] <= visits_a + visits_b
	assert c["pairwise_encounters"] == 2  # exactly the two overlapping bouts


def test_no_cooccupancy_yields_no_rows(monkeypatch):
	"""Only cells a pair met in are stored, so animals that never share a cage store none."""
	rows = [
		stay("A", "cage_1", at(2023, 5, 24, 12, 0, 10), 10),
		stay("B", "cage_2", at(2023, 5, 24, 12, 0, 10), 10),
	]
	assert run_pairwise(monkeypatch, strategies.padded_df_frame(rows, RECORDING)).is_empty()


def test_tunnel_cooccupancy_is_a_meeting(monkeypatch):
	"""Two animals sharing a tunnel meet there, as they would in a cage."""
	# A in c1_c2 [12:00:00, 12:00:10]; B [12:00:05, 12:00:15]; overlap 5 s.
	rows = [
		stay("A", "c1_c2", at(2023, 5, 24, 12, 0, 10), 10),
		stay("B", "c1_c2", at(2023, 5, 24, 12, 0, 15), 10),
	]
	c = pair_cell(
		run_pairwise(monkeypatch, strategies.padded_df_frame(rows, RECORDING)), "A", "B", "tunnel_1"
	)
	assert c["time_together"] == dt.timedelta(seconds=5.0)
	assert c["pairwise_encounters"] == 1


def test_opposite_tunnel_directions_are_one_position(monkeypatch):
	"""Animals passing through one tunnel in opposite directions still meet.

	padded_df carries directional tunnel positions (``c1_c2``/``c2_c1``). Without
	collapsing them first the two animals sit in different positions and the meeting
	vanishes -- so this pins the directionality half of _occupancy_intervals.
	"""
	rows = [
		stay("A", "c1_c2", at(2023, 5, 24, 12, 0, 10), 10),
		stay("B", "c2_c1", at(2023, 5, 24, 12, 0, 15), 10),
	]
	result = run_pairwise(monkeypatch, strategies.padded_df_frame(rows, RECORDING))
	c = pair_cell(result, "A", "B", "tunnel_1")

	assert c["time_together"] == dt.timedelta(seconds=5.0)
	assert c["pairwise_encounters"] == 1


def test_undefined_position_yields_no_meeting(monkeypatch):
	"""``undefined`` is not a place, so overlapping there is not a meeting."""
	rows = [
		stay("A", "undefined", at(2023, 5, 24, 12, 0, 10), 10),
		stay("B", "undefined", at(2023, 5, 24, 12, 0, 15), 10),
	]
	result = run_pairwise(monkeypatch, strategies.padded_df_frame(rows, RECORDING))

	assert result["time_together"].sum() == dt.timedelta(0)
	assert result["pairwise_encounters"].sum() == 0


def bin_cell(result: pl.DataFrame, a: str, b: str, position: str, hour: int) -> dict:
	"""Co-presence totals for one pair, position and hour - not summed over the grid."""
	lo, hi = sorted([a, b])
	sub = result.filter(
		(pl.col("animal_id") == lo)
		& (pl.col("animal_id_2") == hi)
		& (pl.col("position") == position)
		& (pl.col("hour") == hour)
	)
	return {
		"time_together": sub["time_together"].sum(),
		"pairwise_encounters": sub["pairwise_encounters"].sum(),
		"phase_count": set(sub.filter(pl.col("time_together") > dt.timedelta(0))["phase_count"]),
	}


def costay(start, end):
	"""A and B together in cage_1 over [start, end), as one visit each."""
	length = (end - start).total_seconds()
	return [
		{"animal_id": animal, "position": "cage_1", "datetime": end, "time_spent": length}
		for animal in ("A", "B")
	]


def test_meeting_crossing_an_hour_keeps_one_encounter_at_its_start(monkeypatch):
	"""Time is split between the hours it happened in; the meeting is counted once.

	Both halves of the question matter: an hourly time course has to attribute each
	second to its own hour, while an encounter is one event and has to be counted in the
	hour it began, or every meeting that outlives an hour is counted twice.
	"""
	rows = costay(at(2023, 5, 24, 13, 59, 50), at(2023, 5, 24, 14, 0, 10))
	result = run_pairwise(monkeypatch, padded_pieces(rows, RECORDING))

	before = bin_cell(result, "A", "B", "cage_1", hour=13)
	after = bin_cell(result, "A", "B", "cage_1", hour=14)

	assert before["time_together"] == dt.timedelta(seconds=10)
	assert after["time_together"] == dt.timedelta(seconds=10)
	assert before["pairwise_encounters"] == 1
	assert after["pairwise_encounters"] == 0


def test_meeting_crossing_a_phase_is_split_by_the_piece_start(monkeypatch):
	"""The two halves of a meeting over a phase switch belong to different phases.

	Sociability and every phase-level metric read this table, so a second spent in the
	light phase must not arrive labelled dark.
	"""
	# The dark phase begins at 12:00 in this fixture, so the switch is also an hour mark.
	rows = costay(at(2023, 5, 24, 11, 59, 50), at(2023, 5, 24, 12, 0, 10))
	result = run_pairwise(monkeypatch, padded_pieces(rows, RECORDING))

	assert bin_cell(result, "A", "B", "cage_1", hour=11)["phase_count"] == {1}
	assert bin_cell(result, "A", "B", "cage_1", hour=12)["phase_count"] == {2}


def test_time_together_over_bins_sums_to_the_whole_meeting(monkeypatch):
	"""Splitting a meeting across bins conserves it: nothing is lost at a boundary.

	Two hours of continuous co-presence spanning three hour bins, so both ends are
	partial - the shape a dropped or double-counted edge piece would show up in.
	"""
	start, end = at(2023, 5, 24, 13, 30, 0), at(2023, 5, 24, 15, 30, 0)
	result = run_pairwise(monkeypatch, padded_pieces(costay(start, end), RECORDING))
	c = pair_cell(result, "A", "B", "cage_1")

	assert c["time_together"] == end - start
	assert c["pairwise_encounters"] == 1
	assert {
		hour: bin_cell(result, "A", "B", "cage_1", hour)["time_together"] for hour in (13, 14, 15)
	} == {
		13: dt.timedelta(minutes=30),
		14: dt.timedelta(hours=1),
		15: dt.timedelta(minutes=30),
	}
