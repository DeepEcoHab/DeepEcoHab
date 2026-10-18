"""Tests for calculate_tube_test (head-on tunnel encounters).

A tube-test event: the loser enters a tunnel and retreats to the cage it came
from, while the winner enters the same tunnel from the opposite end during an
overlapping interval and exits later. The winner either follows the loser into
the cage it retreated to (CHASE) or returns to its own origin cage (GUARD).

Contact is found by a sweep over tunnel occupancy intervals, so pairing is by real
time overlap and an encounter straddling an hour, phase or day boundary still counts.

calculate_tube_test reads main_df via Recording.load_results (monkeypatched here) and
is called directly. Most cases use the linear layout of
strategies.analysis_cfg, where positions never repeat consecutively per animal so
_resolve_repeat_reads is a no-op; the cases that exercise the repeat-read path use
strategies.ring_cfg, whose antennas sit at the tunnel mouths as in a real setup.
"""

import datetime as dt

import polars as pl
import strategies

from deepecohab.auxiliary_analysis import tube_test
from deepecohab.core.data_model import Recording

RECORDING = strategies.analysis_recording(animal_ids=["A", "B", "C"])
at = strategies.at


def run_tube(
	monkeypatch,
	main_lf,
	recording: Recording | None = None,
	max_dwell: float = 10.0,
	winner_behavior: str = "BOTH",
) -> pl.DataFrame:
	monkeypatch.setattr(Recording, "load_results", lambda self, key, eager=False: main_lf)
	return tube_test.calculate_tube_test(
		recording or RECORDING, max_dwell, winner_behavior
	).collect()


def encounter(
	loser="B",
	winner="A",
	*,
	behavior="chase",
	loser_dwell=2.0,
	winner_dwell=2.0,
	winner_tunnel_exit=(12, 0, 4),
):
	"""Rows for one head-on encounter in tunnel_1 (c1_c2 / c2_c1).

	Loser comes from cage_1, dips into the tunnel and retreats to cage_1. Winner
	comes from cage_2 into the same tunnel; for CHASE it ends in cage_1 (follows
	the loser), for GUARD it returns to cage_2 (its origin).
	"""
	winner_final = "cage_1" if behavior == "chase" else "cage_2"
	we = at(2023, 5, 24, *winner_tunnel_exit)

	return [
		{
			"animal_id": loser,
			"position": "cage_1",
			"datetime": at(2023, 5, 24, 12, 0, 0),
			"time_spent": 1.0,
		},
		{
			"animal_id": winner,
			"position": "cage_2",
			"datetime": at(2023, 5, 24, 12, 0, 0, 500000),
			"time_spent": 1.0,
		},
		# loser tunnel exit at 12:00:03 (entry = exit - dwell)
		{
			"animal_id": loser,
			"position": "c1_c2",
			"datetime": at(2023, 5, 24, 12, 0, 3),
			"time_spent": loser_dwell,
		},
		# winner tunnel exit (entry = exit - dwell)
		{"animal_id": winner, "position": "c2_c1", "datetime": we, "time_spent": winner_dwell},
		{
			"animal_id": loser,
			"position": "cage_1",
			"datetime": at(2023, 5, 24, 12, 0, 5),
			"time_spent": 2.0,
		},
		{
			"animal_id": winner,
			"position": winner_final,
			"datetime": we + dt.timedelta(seconds=2),
			"time_spent": 2.0,
		},
	]


def total(result: pl.DataFrame, winner: str, loser: str) -> int:
	sub = result.filter((pl.col("winner") == winner) & (pl.col("loser") == loser))
	return int(sub["tube_test"].sum())


def test_chase_event_detected(monkeypatch):
	"""A head-on CHASE encounter is counted once, with the retreater as the loser."""
	result = run_tube(monkeypatch, strategies.main_df_frame(encounter(behavior="chase"), RECORDING))
	assert total(result, "A", "B") == 1
	assert total(result, "B", "A") == 0  # direction guard, no double count


def test_winner_behavior_isolates_chase(monkeypatch):
	"""A CHASE encounter counts under CHASE/BOTH but not under GUARD."""
	rows = strategies.main_df_frame(encounter(behavior="chase"), RECORDING)
	assert run_tube(monkeypatch, rows, winner_behavior="CHASE")["tube_test"].sum() == 1
	assert run_tube(monkeypatch, rows, winner_behavior="BOTH")["tube_test"].sum() == 1
	assert run_tube(monkeypatch, rows, winner_behavior="GUARD")["tube_test"].sum() == 0


def test_winner_behavior_isolates_guard(monkeypatch):
	"""A GUARD encounter counts under GUARD/BOTH but not under CHASE."""
	rows = strategies.main_df_frame(encounter(behavior="guard"), RECORDING)
	assert run_tube(monkeypatch, rows, winner_behavior="GUARD")["tube_test"].sum() == 1
	assert run_tube(monkeypatch, rows, winner_behavior="BOTH")["tube_test"].sum() == 1
	assert run_tube(monkeypatch, rows, winner_behavior="CHASE")["tube_test"].sum() == 0


def test_max_dwell_excludes_inflated_segment(monkeypatch):
	"""An inflated loser tunnel dwell is dropped at the default cap but kept if raised."""
	rows = strategies.main_df_frame(encounter(behavior="chase", loser_dwell=20.0), RECORDING)
	# Default max_dwell=10 drops the 20 s loser segment...
	assert run_tube(monkeypatch, rows)["tube_test"].sum() == 0
	# ...raising the cap past it restores the (still overlapping) encounter.
	assert run_tube(monkeypatch, rows, max_dwell=30.0)["tube_test"].sum() == 1


def test_non_overlapping_intervals_not_an_event(monkeypatch):
	"""If the two tunnel intervals do not overlap in time, it is not a tube test."""
	# Winner enters the tunnel only after the loser has already exited it.
	rows = strategies.main_df_frame(
		encounter(behavior="chase", winner_dwell=2.0, winner_tunnel_exit=(12, 0, 7)), RECORDING
	)
	# loser tunnel interval [12:00:01, 12:00:03]; winner [12:00:05, 12:00:07] -> no overlap.
	assert run_tube(monkeypatch, rows)["tube_test"].sum() == 0


def test_same_origin_is_not_head_on(monkeypatch):
	"""Two animals entering the tunnel from the same cage is not a head-on encounter."""
	rows = [
		{
			"animal_id": "B",
			"position": "cage_1",
			"datetime": at(2023, 5, 24, 12, 0, 0),
			"time_spent": 1.0,
		},
		{
			"animal_id": "A",
			"position": "cage_1",
			"datetime": at(2023, 5, 24, 12, 0, 0, 500000),
			"time_spent": 1.0,
		},
		{
			"animal_id": "B",
			"position": "c1_c2",
			"datetime": at(2023, 5, 24, 12, 0, 3),
			"time_spent": 2.0,
		},
		{
			"animal_id": "A",
			"position": "c1_c2",
			"datetime": at(2023, 5, 24, 12, 0, 4),
			"time_spent": 2.0,
		},
		{
			"animal_id": "B",
			"position": "cage_1",
			"datetime": at(2023, 5, 24, 12, 0, 5),
			"time_spent": 2.0,
		},
		{
			"animal_id": "A",
			"position": "cage_1",
			"datetime": at(2023, 5, 24, 12, 0, 6),
			"time_spent": 2.0,
		},
	]
	assert run_tube(monkeypatch, strategies.main_df_frame(rows, RECORDING))["tube_test"].sum() == 0


def test_event_carries_tunnel_position(monkeypatch):
	"""The counted event is attributed to the tunnel it occurred in (tunnel_1)."""
	result = run_tube(monkeypatch, strategies.main_df_frame(encounter(behavior="chase"), RECORDING))

	# Grid is reindexed over the undirected tunnels, so every tunnel is present...
	assert set(result["position"].cast(pl.String).unique()) == set(RECORDING.layout.tunnel_names)
	# ...but the single A-beats-B event sits in tunnel_1 (c1_c2 / c2_c1).
	event = result.filter(pl.col("tube_test") > 0)
	assert event.height == 1
	assert event["position"].cast(pl.String).to_list() == ["tunnel_1"]


def test_no_events_yields_all_zero_grid(monkeypatch):
	"""No head-on encounters (a lone retreating animal) -> a full grid of zeros."""
	rows = [
		{
			"animal_id": "B",
			"position": "cage_1",
			"datetime": at(2023, 5, 24, 12, 0, 0),
			"time_spent": 1.0,
		},
		{
			"animal_id": "B",
			"position": "c1_c2",
			"datetime": at(2023, 5, 24, 12, 0, 3),
			"time_spent": 2.0,
		},
		{
			"animal_id": "B",
			"position": "cage_1",
			"datetime": at(2023, 5, 24, 12, 0, 5),
			"time_spent": 2.0,
		},
	]
	result = run_tube(monkeypatch, strategies.main_df_frame(rows, RECORDING))
	assert result.height > 0
	assert result["tube_test"].sum() == 0
	assert result["tube_test"].null_count() == 0


# --- the repeat-read path ----------------------------------------------------
# These use the real ring geometry (antennas at the tunnel mouths), which is the
# only layout where a repeat read carries the in-and-out meaning the tube test
# depends on. The linear RECORDING above cannot express a retreat at all.

RING = strategies.ring_recording(animal_ids=["A", "B", "C"])


def T(second: int, microsecond: int = 0):
	"""A day-1 light-phase timestamp, `second` seconds past 12:00."""
	return at(2023, 5, 24, 12, 0, second, microsecond)


def visit(animal: str, position: str, end, dwell: float) -> dict:
	"""One main_df row: `animal` held `position` for `dwell` s and left at `end`."""
	return {"animal_id": animal, "position": position, "datetime": end, "time_spent": dwell}


def encounter_at(loser_exit, winner_exit) -> list[dict]:
	"""One head-on CHASE encounter in tunnel_1, anchored at the two tunnel exits.

	B dips in from cage_1 and retreats there; A comes the other way from cage_2 and
	follows B into cage_1. The two tunnel intervals overlap as long as the exits are
	less than the 2 s dwell apart.
	"""
	return [
		visit("B", "cage_1", loser_exit - dt.timedelta(seconds=3), 1.0),
		visit("A", "cage_2", loser_exit - dt.timedelta(seconds=2), 1.0),
		visit("B", "c1_c2", loser_exit, 2.0),
		visit("A", "c2_c1", winner_exit, 2.0),
		visit("B", "cage_1", loser_exit + dt.timedelta(seconds=2), 2.0),
		visit("A", "cage_1", winner_exit + dt.timedelta(seconds=2), 2.0),
	]


def test_tunnel_entry_map_derived_from_shipped_layouts():
	"""The antenna -> tunnel map comes from the config, covering every shipped layout."""
	default = tube_test._tunnel_entry_by_antenna(strategies.ring_recording(section="default"))
	field = tube_test._tunnel_entry_by_antenna(strategies.ring_recording(section="field"))

	assert len(default) == 8
	assert default["1"] == "c1_c2"  # antenna 1 is the cage_1 mouth of tunnel_1
	assert default["8"] == "c1_c4"  # antenna 8 is the cage_1 mouth of tunnel_4
	assert len(field) == 16
	assert field["9"] == "cE_cF"  # 16-antenna layout, well past the old hardcoded 8


def test_linear_layout_skips_ambiguous_antennas():
	"""An antenna that leads into several tunnels is left out, so the rewrite no-ops."""
	# In the linear fixture an antenna sits inside a cage, so antennas 2 and 3 each lead
	# into two tunnels and a repeat read there carries no in-and-out meaning.
	assert set(tube_test._tunnel_entry_by_antenna(RECORDING)) == {"1", "4"}


def test_repeat_read_relabels_only_the_return_crossing():
	"""In a run of reads on one antenna, every second row becomes tunnel occupancy."""
	# B crosses into cage_1 at antenna 8, walks to antenna 1, pokes into tunnel_1 and backs
	# out over antenna 1 again, then leaves cage_1 over antenna 8.
	reads = [(7, T(0)), (8, T(1)), (1, T(2)), (1, T(5)), (8, T(9))]
	lf = strategies.main_df_frame(strategies.antenna_reads("B", reads, RING), RING)

	positions = (
		tube_test._resolve_repeat_reads(lf.sort("datetime"), RING)
		.collect()["position"]
		.cast(pl.String)
		.to_list()
	)
	# Only the second antenna-1 row -- the return crossing -- moves into the tunnel.
	assert positions == ["undefined", "c4_c1", "cage_1", "c1_c2", "cage_1"]


def test_field_layout_never_leaks_a_raw_antenna_into_position():
	"""A 16-antenna layout produces only positions that layout actually defines.

	Regression: the map used to be a hardcoded 8-antenna dict applied with `replace`, so
	antenna 9 passed through as the bare string "9" and antenna 3 became "c2_c3" -- a
	default-layout tunnel absent from the field config. Both then failed the tunnel filter,
	losing every field retreat, and corrupted their neighbours' prev/next positions.
	"""
	field = strategies.ring_recording(animal_ids=["A", "B", "C"], section="field")
	reads = [(7, T(0)), (8, T(1)), (9, T(2)), (9, T(5)), (8, T(9))]
	lf = strategies.main_df_frame(strategies.antenna_reads("A", reads, field), field)

	produced = set(
		tube_test._resolve_repeat_reads(lf.sort("datetime"), field)
		.collect()["position"]
		.cast(pl.String)
	)

	legal = set(field.layout.positions_non_directional) | set(field.layout.tunnels_map)
	assert produced <= legal, f"positions outside the layout: {sorted(produced - legal)}"
	assert "cE_cF" in produced  # the return crossing at antenna 9, named as the field maps it


def test_retreat_seen_only_as_a_repeat_read_is_counted(monkeypatch):
	"""A head-on encounter whose retreats exist only as repeat reads is detected."""
	# B pokes into tunnel_1 from cage_1 and backs out; A does the same from cage_2 but holds
	# the tunnel longer, so B loses. Neither retreat reaches the far antenna, so both are
	# repeat reads and nothing but _resolve_repeat_reads can see them.
	rows = strategies.antenna_reads(
		"B", [(7, T(0)), (8, T(1)), (1, T(2)), (1, T(5)), (8, T(9))], RING
	)
	rows += strategies.antenna_reads(
		"A", [(4, T(0, 500000)), (3, T(1, 500000)), (2, T(3)), (2, T(8)), (3, T(11))], RING
	)
	lf = strategies.main_df_frame(rows, RING)

	assert total(run_tube(monkeypatch, lf, recording=RING), "A", "B") == 1

	# Without the relabelling both retreats stay cage rows, nobody is ever in the tunnel and
	# the encounter vanishes -- which is why the helper is load-bearing rather than cosmetic.
	monkeypatch.setattr(tube_test, "_resolve_repeat_reads", lambda lf, cfg: lf)
	assert run_tube(monkeypatch, lf, recording=RING)["tube_test"].sum() == 0


# --- boundaries and encounter identity ---------------------------------------


def test_encounter_across_phase_boundary_is_counted(monkeypatch):
	"""An encounter straddling the light -> dark switch is counted, not dropped.

	Regression: pairing used to be a self-join keyed on phase/day/phase_count, so the two
	halves of a boundary-straddling encounter landed in different groups and never met.
	"""
	inside = encounter_at(T(3), T(4))
	straddle = encounter_at(at(2023, 5, 24, 19, 59, 59, 500000), at(2023, 5, 24, 20, 0, 0, 500000))

	assert (
		run_tube(monkeypatch, strategies.main_df_frame(inside, RECORDING))["tube_test"].sum() == 1
	)
	assert (
		run_tube(monkeypatch, strategies.main_df_frame(straddle, RECORDING))["tube_test"].sum() == 1
	)


def test_encounter_across_midnight_is_counted(monkeypatch):
	"""The same holds across a day change inside one dark phase."""
	rows = encounter_at(at(2023, 5, 24, 23, 59, 59, 500000), at(2023, 5, 25, 0, 0, 0, 500000))
	result = run_tube(monkeypatch, strategies.main_df_frame(rows, RECORDING))

	assert total(result, "A", "B") == 1
	# Binned on the winner's exit, so the whole event lands on day 2 rather than splitting.
	assert result.filter(pl.col("tube_test") > 0)["day"].to_list() == [2]


def test_continuous_contact_counts_once(monkeypatch):
	"""One unbroken co-presence is one encounter, however many sweep spans it spans."""
	rows = [
		visit("A", "cage_2", T(0), 1.0),
		visit("B", "cage_1", T(1), 1.0),
		visit("A", "c2_c1", T(10), 10.0),  # A holds tunnel_1 across the whole contact
		visit("B", "c1_c2", T(4), 3.0),  # B dips in [1, 4] and retreats
		visit("B", "cage_1", T(6), 2.0),
		visit("A", "cage_2", T(12), 2.0),
	]
	assert total(run_tube(monkeypatch, strategies.main_df_frame(rows, RECORDING)), "A", "B") == 1


def test_separated_challenges_count_twice(monkeypatch):
	"""Two pokes with a step back into the cage between them are two encounters."""
	rows = [
		visit("A", "cage_2", T(0), 1.0),
		visit("B", "cage_1", T(1), 1.0),
		visit("B", "c1_c2", T(2), 1.0),  # first challenge [1, 2]
		visit("B", "cage_1", T(3), 1.0),  # B steps out, breaking co-presence
		visit("B", "c1_c2", T(4), 1.0),  # second challenge [3, 4]
		visit("A", "c2_c1", T(10), 10.0),  # A holds the tunnel throughout
		visit("B", "cage_1", T(6), 2.0),
		visit("A", "cage_2", T(12), 2.0),
	]
	assert total(run_tube(monkeypatch, strategies.main_df_frame(rows, RECORDING)), "A", "B") == 2


def test_mutual_retreat_scores_the_later_retreater_as_winner(monkeypatch):
	"""When both animals back out, the one that left the tunnel first is the loser."""
	rows = [
		visit("B", "cage_1", T(0), 1.0),
		visit("A", "cage_2", T(1), 1.0),
		visit("B", "c1_c2", T(3), 3.0),  # B in the tunnel [0, 3], back to cage_1
		visit("A", "c2_c1", T(4), 3.0),  # A in the tunnel [1, 4], back to cage_2
		visit("B", "cage_1", T(5), 2.0),
		visit("A", "cage_2", T(6), 2.0),
	]
	result = run_tube(monkeypatch, strategies.main_df_frame(rows, RECORDING))

	assert total(result, "A", "B") == 1  # B left first, so A holds the tunnel
	assert total(result, "B", "A") == 0  # and the pair is never scored both ways
