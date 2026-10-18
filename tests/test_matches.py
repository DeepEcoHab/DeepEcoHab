"""Tests for calculate_matches (event-level chasing detection).

A chasing event is a loser exiting a tunnel and a winner who entered that same
tunnel from a cage 0.1-1.2 s earlier, exiting after the loser. calculate_matches
reads main_df via Recording.load_results, so we monkeypatch that to feed a hand-built
main_df and call the step directly, as test_ranking does.

main_df rows mark the position an animal *left* and when it left, so a tunnel row
is a tunnel *exit* and its (shifted) predecessor supplies the tunnel entry time
and the cage it came from.
"""

import datetime as dt

import polars as pl
import strategies

from deepecohab.core import antenna_analysis
from deepecohab.core.data_model import AnalysisParams, Recording

RECORDING = strategies.analysis_recording(animal_ids=["A", "B", "C"])
at = strategies.at


def run_matches(monkeypatch, main_lf, **kwargs) -> pl.DataFrame:
	"""Call calculate_matches' compute body with main_df injected via _get_data."""
	monkeypatch.setattr(Recording, "load_results", lambda self, key, eager=False: main_lf)
	return antenna_analysis.calculate_matches(RECORDING, AnalysisParams(**kwargs)).collect()


def chase(
	winner, loser, *, entry, exit_gap, winner_exit_gap=5.0, tunnel="c1_c2", from_cage="cage_1"
):
	"""Rows for a single chase through ``tunnel``.

	Winner enters the tunnel from ``from_cage`` at ``entry``; loser exits the tunnel
	``exit_gap`` seconds later; winner exits ``winner_exit_gap`` seconds after that.
	"""
	loser_exit = entry + dt.timedelta(seconds=exit_gap)
	winner_exit = loser_exit + dt.timedelta(seconds=winner_exit_gap)
	return [
		{"animal_id": winner, "position": from_cage, "datetime": entry, "time_spent": 5.0},
		{"animal_id": loser, "position": tunnel, "datetime": loser_exit, "time_spent": 1.0},
		{"animal_id": winner, "position": tunnel, "datetime": winner_exit, "time_spent": 1.0},
	]


def test_output_schema(monkeypatch):
	"""Result carries the grid columns plus winner/loser/datetime/chasing_length."""
	rows = chase("A", "B", entry=at(2023, 5, 24, 12, 0, 0), exit_gap=0.5)
	result = run_matches(monkeypatch, strategies.main_df_frame(rows, RECORDING))
	assert set(result.columns) == {
		"phase",
		"day",
		"phase_count",
		"hour",
		"position",
		"winner",
		"loser",
		"datetime",
		"chasing_length",
	}


def test_genuine_chase_detected(monkeypatch):
	"""A follow-through within the window yields exactly one winner/loser row."""
	rows = chase("A", "B", entry=at(2023, 5, 24, 12, 0, 0), exit_gap=0.5)
	result = run_matches(monkeypatch, strategies.main_df_frame(rows, RECORDING))

	assert result.height == 1
	row = result.row(0, named=True)
	assert row["winner"] == "A"
	assert row["loser"] == "B"
	assert row["position"] == "c1_c2"
	# chasing_length is the gap between the two exits, not the winner's entry to the
	# loser's exit: the loser leaves at entry+0.5 and the winner 5 s behind it.
	assert row["chasing_length"] == dt.timedelta(seconds=5.0)


def test_no_double_count_for_symmetric_pair(monkeypatch):
	"""Both animals enter the tunnel from a cage, so both are winner candidates.

	Each one's exit is also a loser query, so the mutual case is the one that could
	count the pass twice - once in each direction. Only the animal that leaves last can
	have been following, which is what ``loser_exit < winner_exit`` enforces.
	"""
	entry = at(2023, 5, 24, 12, 0, 0)
	rows = [
		{"animal_id": "A", "position": "cage_1", "datetime": entry, "time_spent": 5.0},
		{"animal_id": "B", "position": "cage_1", "datetime": entry, "time_spent": 5.0},
		{
			"animal_id": "B",
			"position": "c1_c2",
			"datetime": entry + dt.timedelta(seconds=0.5),
			"time_spent": 0.5,
		},
		{
			"animal_id": "A",
			"position": "c1_c2",
			"datetime": entry + dt.timedelta(seconds=5),
			"time_spent": 5.0,
		},
	]
	result = run_matches(monkeypatch, strategies.main_df_frame(rows, RECORDING))

	assert result.height == 1
	row = result.row(0, named=True)
	assert (row["winner"], row["loser"]) == ("A", "B")
	assert row["chasing_length"] == dt.timedelta(seconds=4.5)


def test_window_endpoints_are_exclusive(monkeypatch):
	"""closed="none": gaps exactly at 0.1 s and 1.2 s do not count."""
	for gap in (0.1, 1.2):
		rows = chase("A", "B", entry=at(2023, 5, 24, 12, 0, 0), exit_gap=gap)
		result = run_matches(monkeypatch, strategies.main_df_frame(rows, RECORDING))
		assert result.height == 0, f"gap {gap} should be excluded"


def test_inside_window_counts(monkeypatch):
	"""Gaps just inside either end of the window do count."""
	for gap in (0.11, 1.19):
		rows = chase("A", "B", entry=at(2023, 5, 24, 12, 0, 0), exit_gap=gap)
		result = run_matches(monkeypatch, strategies.main_df_frame(rows, RECORDING))
		assert result.height == 1, f"gap {gap} should be included"


def test_too_fast_and_too_slow_excluded(monkeypatch):
	"""Followers arriving too soon (<0.1 s) or too late (>1.2 s) are not chases."""
	for gap in (0.05, 2.0):
		rows = chase("A", "B", entry=at(2023, 5, 24, 12, 0, 0), exit_gap=gap)
		result = run_matches(monkeypatch, strategies.main_df_frame(rows, RECORDING))
		assert result.height == 0, f"gap {gap} should be excluded"


def test_winner_must_enter_from_a_cage(monkeypatch):
	"""If the winner entered the tunnel from another tunnel (not a cage), no chase."""
	# Winner's predecessor position is a tunnel, so prev_position is not in cages.
	rows = [
		{
			"animal_id": "A",
			"position": "c2_c3",
			"datetime": at(2023, 5, 24, 12, 0, 0),
			"time_spent": 1.0,
		},
		{
			"animal_id": "B",
			"position": "c1_c2",
			"datetime": at(2023, 5, 24, 12, 0, 0, 500000),
			"time_spent": 1.0,
		},
		{
			"animal_id": "A",
			"position": "c1_c2",
			"datetime": at(2023, 5, 24, 12, 0, 5),
			"time_spent": 1.0,
		},
	]
	result = run_matches(monkeypatch, strategies.main_df_frame(rows, RECORDING))
	assert result.height == 0


def test_different_tunnels_not_a_chase(monkeypatch):
	"""Winner and loser must traverse the same tunnel."""
	rows = [
		{
			"animal_id": "A",
			"position": "cage_1",
			"datetime": at(2023, 5, 24, 12, 0, 0),
			"time_spent": 5.0,
		},
		{
			"animal_id": "B",
			"position": "c2_c3",
			"datetime": at(2023, 5, 24, 12, 0, 0, 500000),
			"time_spent": 1.0,
		},
		{
			"animal_id": "A",
			"position": "c1_c2",
			"datetime": at(2023, 5, 24, 12, 0, 5),
			"time_spent": 1.0,
		},
	]
	result = run_matches(monkeypatch, strategies.main_df_frame(rows, RECORDING))
	assert result.height == 0


def test_chase_straddling_hour_boundary_assigned_to_chaser_hour(monkeypatch):
	"""A chase whose tunnel exits fall in different hour bins is still detected and
	assigned to the chaser's (winner's, later) hour.

	The join blocks on the tunnel rather than the hour bucket, so a genuine
	follow-through is not split across two hour bins and dropped; its grid columns
	are derived from the winner's exit timestamp.
	"""
	# Winner enters from cage_1 at 12:59:59.5 (hour 12); loser exits the tunnel at
	# 12:59:59.9 (0.4 s later, hour 12) but the winner exits at 13:00:01 (hour 13).
	rows = [
		{
			"animal_id": "A",
			"position": "cage_1",
			"datetime": at(2023, 5, 24, 12, 59, 59, 500000),
			"time_spent": 5.0,
		},
		{
			"animal_id": "B",
			"position": "c1_c2",
			"datetime": at(2023, 5, 24, 12, 59, 59, 900000),
			"time_spent": 1.0,
		},
		{
			"animal_id": "A",
			"position": "c1_c2",
			"datetime": at(2023, 5, 24, 13, 0, 1),
			"time_spent": 1.0,
		},
	]
	result = run_matches(monkeypatch, strategies.main_df_frame(rows, RECORDING))
	assert result.height == 1
	row = result.row(0, named=True)
	assert row["winner"] == "A"
	assert row["loser"] == "B"
	assert row["hour"] == 13  # chaser's (winner's) exit hour, not the loser's hour 12
	assert row["chasing_length"] == dt.timedelta(seconds=1.1)  # 12:59:59.9 to 13:00:01


def test_recording_gap_fabricates_no_chase(monkeypatch):
	"""A long break between a winner's entry and a later tunnel exit is not a chase."""
	# Winner enters the tunnel, then a multi-hour gap, then the loser exits: the
	# entry->exit gap is far outside the chasing window.
	rows = [
		{
			"animal_id": "A",
			"position": "cage_1",
			"datetime": at(2023, 5, 24, 12, 0, 0),
			"time_spent": 5.0,
		},
		{
			"animal_id": "A",
			"position": "c1_c2",
			"datetime": at(2023, 5, 24, 12, 0, 1),
			"time_spent": 1.0,
		},
		{
			"animal_id": "B",
			"position": "c1_c2",
			"datetime": at(2023, 5, 24, 15, 0, 0),
			"time_spent": 1.0,
		},
	]
	result = run_matches(monkeypatch, strategies.main_df_frame(rows, RECORDING))
	assert result.height == 0


def test_no_qualifying_events_returns_empty(monkeypatch):
	"""Only cage visits (no tunnel follow-throughs) yield an empty, typed frame."""
	rows = [
		{
			"animal_id": "A",
			"position": "cage_1",
			"datetime": at(2023, 5, 24, 12, 0, 0),
			"time_spent": 5.0,
		},
		{
			"animal_id": "B",
			"position": "cage_2",
			"datetime": at(2023, 5, 24, 12, 0, 1),
			"time_spent": 5.0,
		},
	]
	result = run_matches(monkeypatch, strategies.main_df_frame(rows, RECORDING))
	assert result.height == 0
	assert "winner" in result.columns
	assert "loser" in result.columns


def at_offset(seconds: float) -> dt.datetime:
	"""A moment inside day-1 hour 12, ``seconds`` after the hour."""
	return at(2023, 5, 24, 12, 0, 0) + dt.timedelta(seconds=seconds)


def test_two_winners_behind_one_loser_are_two_events(monkeypatch):
	"""Both followers are inside the tunnel at the exit, so both bits decode.

	The mask at a loser's exit is a set, not a single winner: a chase down a tunnel by
	two animals is two dominance events, and dropping either would silently favour
	whichever one the join happened to see first.
	"""
	rows = [
		{"animal_id": "A", "position": "cage_1", "datetime": at_offset(0), "time_spent": 5.0},
		{"animal_id": "C", "position": "cage_1", "datetime": at_offset(0.1), "time_spent": 5.0},
		{"animal_id": "B", "position": "c1_c2", "datetime": at_offset(0.6), "time_spent": 1.0},
		{"animal_id": "A", "position": "c1_c2", "datetime": at_offset(5), "time_spent": 5.0},
		{"animal_id": "C", "position": "c1_c2", "datetime": at_offset(6), "time_spent": 6.0},
	]
	result = run_matches(monkeypatch, strategies.main_df_frame(rows, RECORDING))

	assert result.height == 2
	assert set(result["winner"]) == {"A", "C"}
	assert set(result["loser"]) == {"B"}


def test_two_losers_ahead_of_one_winner_are_two_events(monkeypatch):
	"""One winner overtaking two animals is two events, one per loser exit.

	Each exit is its own query against the same open winner pass, so the pass has to be
	recoverable more than once.
	"""
	rows = [
		{"animal_id": "A", "position": "cage_1", "datetime": at_offset(0), "time_spent": 5.0},
		{"animal_id": "B", "position": "c1_c2", "datetime": at_offset(0.5), "time_spent": 1.0},
		{"animal_id": "C", "position": "c1_c2", "datetime": at_offset(0.9), "time_spent": 1.0},
		{"animal_id": "A", "position": "c1_c2", "datetime": at_offset(5), "time_spent": 5.0},
	]
	result = run_matches(monkeypatch, strategies.main_df_frame(rows, RECORDING))

	assert result.height == 2
	assert set(result["winner"]) == {"A"}
	assert set(result["loser"]) == {"B", "C"}


def test_repeat_pass_matches_the_open_one_not_the_first(monkeypatch):
	"""A winner that has used the tunnel before is timed from its current entry.

	The as-of join takes the latest entry at or before the loser's exit. Pairing the
	loser with the animal's *first* pass instead would measure a ten-second follow
	through and throw the event away.
	"""
	rows = [
		# First pass through c1_c2 and back, then a second pass ten seconds later.
		{"animal_id": "A", "position": "cage_1", "datetime": at_offset(0), "time_spent": 5.0},
		{"animal_id": "A", "position": "c1_c2", "datetime": at_offset(1), "time_spent": 1.0},
		{"animal_id": "A", "position": "cage_2", "datetime": at_offset(5), "time_spent": 4.0},
		{"animal_id": "A", "position": "c2_c1", "datetime": at_offset(6), "time_spent": 1.0},
		{"animal_id": "A", "position": "cage_1", "datetime": at_offset(10), "time_spent": 4.0},
		{"animal_id": "B", "position": "c1_c2", "datetime": at_offset(10.5), "time_spent": 1.0},
		{"animal_id": "A", "position": "c1_c2", "datetime": at_offset(12), "time_spent": 2.0},
	]
	result = run_matches(monkeypatch, strategies.main_df_frame(rows, RECORDING))

	assert result.height == 1
	row = result.row(0, named=True)
	assert (row["winner"], row["loser"]) == ("A", "B")
	assert row["chasing_length"] == dt.timedelta(seconds=1.5)


def test_loser_arriving_from_another_tunnel_still_counts(monkeypatch):
	"""Only the winner has to be seen entering from a cage.

	The loser's own entry is often the read that goes missing - it is the animal being
	pushed through - so requiring a cage behind it too would discard the clearest
	chases.
	"""
	rows = [
		{"animal_id": "B", "position": "c2_c1", "datetime": at_offset(-1), "time_spent": 1.0},
		{"animal_id": "A", "position": "cage_1", "datetime": at_offset(0), "time_spent": 5.0},
		{"animal_id": "B", "position": "c1_c2", "datetime": at_offset(0.5), "time_spent": 1.5},
		{"animal_id": "A", "position": "c1_c2", "datetime": at_offset(5), "time_spent": 5.0},
	]
	result = run_matches(monkeypatch, strategies.main_df_frame(rows, RECORDING))

	assert result.height == 1
	assert result.row(0, named=True)["winner"] == "A"
	assert result.row(0, named=True)["loser"] == "B"


def test_head_on_passes_are_never_a_chase(monkeypatch):
	"""Two animals crossing the same tunnel in opposite directions did not chase.

	The tunnel keeps its direction here, unlike in the occupancy tables: ``c1_c2`` and
	``c2_c1`` are different positions, so neither animal is ever a candidate for the
	other's exit.
	"""
	rows = [
		{"animal_id": "B", "position": "cage_2", "datetime": at_offset(-1), "time_spent": 5.0},
		{"animal_id": "A", "position": "cage_1", "datetime": at_offset(0), "time_spent": 5.0},
		{"animal_id": "B", "position": "c2_c1", "datetime": at_offset(0.5), "time_spent": 1.5},
		{"animal_id": "A", "position": "c1_c2", "datetime": at_offset(5), "time_spent": 5.0},
	]
	result = run_matches(monkeypatch, strategies.main_df_frame(rows, RECORDING))

	assert result.height == 0
