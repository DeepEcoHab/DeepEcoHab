"""Tests for calculate_ranking (Plackett-Luce dominance ranking).

calculate_ranking replays each chasing event in match_df as a one-on-one match
and emits the full rating trajectory (one row per animal after every match). The
step reads match_df via Recording.load_results, so we monkeypatch that to feed a small,
hand-built match table and exercise the pure compute body directly via
``__wrapped__`` .
"""

import datetime as dt
import random
from functools import partial
from zoneinfo import ZoneInfo

import polars as pl
import pytest
import strategies

from deepecohab.core import antenna_analysis, ranking_algorithm
from deepecohab.core.data_model import AnalysisParams, Recording

TZ = ZoneInfo("Europe/Warsaw")

PHASE_CFG = {"light_phase": dt.time(7, 0), "dark_phase": dt.time(20, 0)}

MODEL = ranking_algorithm.MODEL


at = partial(strategies.at, tz=str(TZ))


def expected_ordinal(mu: float | None = None, sigma: float | None = None) -> float:
	"""Ordinal a never-playing animal should hold, rounded as the step rounds it."""
	return round(MODEL.rating(mu=mu, sigma=sigma).ordinal(), 3)


def make_match_df(rows: list[tuple[str, str, dt.datetime]], recording) -> pl.LazyFrame:
	"""Build a match_df LazyFrame from (loser, winner, datetime) tuples.

	The ranking carries the calendar columns of each match through, so the frame has
	to be a full match_df rather than just the three columns the replay reads.
	"""
	return strategies.match_df_frame(
		[
			{"loser": loser, "winner": winner, "position": "c1_c2", "datetime": moment}
			for loser, winner, moment in rows
		],
		recording,
	)


def run_ranking(
	monkeypatch, matches, animal_ids, prev_ranking=None, root=None, **params
) -> pl.DataFrame:
	"""Call the pure ranking body with match_df injected in place of _get_data.

	``prev_ranking`` is stored with the recording first, which needs a ``root`` folder.
	"""
	monkeypatch.setattr(Recording, "load_results", lambda self, key, eager=False: match_df)
	# The window spans the match datetimes below, so phase_count resolves off the grid.
	recording = strategies.analysis_recording(
		animal_ids=animal_ids,
		tz=str(TZ),
		start="2023-05-24 00:00:00",
		finish="2023-05-26 23:00:00",
		phases=PHASE_CFG,
		root=root,
	)
	match_df = make_match_df(matches, recording)
	monkeypatch.setattr(Recording, "load_results", lambda self, key, eager=False: match_df)

	if prev_ranking is not None:
		recording.set_prev_ranking(prev_ranking)
	# A recording without a folder has no stored previous ranking to look for.
	params.setdefault("use_prev_ranking", root is not None)
	return antenna_analysis.calculate_ranking(recording, AnalysisParams(**params)).collect()


def final_block(result: pl.DataFrame) -> dict[str, float]:
	"""Map animal_id -> ordinal in the last-emitted block (final standings)."""
	last = result.filter(pl.col("datetime") == result["datetime"].max())
	return dict(last.select("animal_id", "ordinal").iter_rows())


def test_output_schema(monkeypatch):
	"""Result carries the rating columns plus the derived phase/day/hour."""
	result = run_ranking(monkeypatch, [("B", "A", at(2023, 5, 24, 12, 0, 0))], ["A", "B"])
	assert set(result.columns) == {
		"animal_id",
		"mu",
		"sigma",
		"ordinal",
		"datetime",
		"phase",
		"day",
		"phase_count",
		"hour",
		"social_rank",
	}


def test_phase_count_aligned_to_grid(monkeypatch):
	"""phase_count is resolved off the dense grid, counting phases globally.

	The experiment starts at the 07:00 light onset, so light is phase #1 and the 21:00
	dark-phase event falls in phase #2. Numbering this one-row frame on its own would
	call it phase #1, so the answer only comes out right off the grid.
	"""
	result = run_ranking(monkeypatch, [("B", "A", at(2023, 5, 24, 21, 0, 0))], ["A", "B"])
	assert result["phase_count"].null_count() == 0
	assert set(result["phase_count"].unique().to_list()) == {2}


def test_one_row_per_animal_per_match(monkeypatch):
	"""Every match emits one row for every animal (full trajectory)."""
	matches = [
		("B", "A", at(2023, 5, 24, 12, 0, 0)),
		("B", "A", at(2023, 5, 24, 12, 1, 0)),
		("B", "A", at(2023, 5, 24, 12, 2, 0)),
	]
	result = run_ranking(monkeypatch, matches, ["A", "B", "C"])
	assert result.height == len(matches) * 3  # 3 animals


def test_consistent_winner_outranks_loser(monkeypatch):
	"""If A always beats B, A's final ordinal exceeds B's."""
	matches = [("B", "A", at(2023, 5, 24, 12, i, 0)) for i in range(5)]
	result = run_ranking(monkeypatch, matches, ["A", "B"])

	ordinals = final_block(result)
	assert ordinals["A"] > ordinals["B"]


def test_uninvolved_animal_keeps_default_ordinal(monkeypatch):
	"""An animal that never plays stays at the model's default rating."""
	matches = [("B", "A", at(2023, 5, 24, 12, 0, 0))]
	result = run_ranking(monkeypatch, matches, ["A", "B", "C"])

	c_rows = result.filter(pl.col("animal_id") == "C")
	assert (c_rows["ordinal"] == expected_ordinal()).all()


def test_winner_and_loser_move_in_opposite_directions(monkeypatch):
	"""After one match the winner's ordinal rises above default and the loser's falls below."""
	result = run_ranking(monkeypatch, [("B", "A", at(2023, 5, 24, 12, 0, 0))], ["A", "B"])

	ordinals = final_block(result)
	assert ordinals["A"] > expected_ordinal()
	assert ordinals["B"] < expected_ordinal()


# --- social_rank ------------------------------------------------------------


def test_social_rank_labels_each_snapshot(monkeypatch):
	"""Per match, the top ordinal is dominant, the bottom subordinate, the rest middle."""
	matches = [
		("B", "A", at(2023, 5, 24, 12, 0, 0)),
		("C", "B", at(2023, 5, 24, 12, 1, 0)),
	]
	result = run_ranking(monkeypatch, matches, ["A", "B", "C", "D"])

	for (_,), block in result.group_by("datetime"):
		top = block.filter(pl.col("social_rank") == "dominant")
		bottom = block.filter(pl.col("social_rank") == "subordinate")

		assert top["ordinal"].to_list() == [block["ordinal"].max()]
		assert bottom["ordinal"].to_list() == [block["ordinal"].min()]
		assert (block["social_rank"] == "middle").sum() == 2


def test_social_rank_pair_has_no_middle(monkeypatch):
	"""Two animals leave the middle category unused, not misassigned."""
	result = run_ranking(monkeypatch, [("B", "A", at(2023, 5, 24, 12, 0, 0))], ["A", "B"])

	assert dict(result.select("animal_id", "social_rank").iter_rows()) == {
		"A": "dominant",
		"B": "subordinate",
	}


# --- prev_ranking continuation (regression: this path used to crash) ---------


@pytest.mark.parametrize("as_lazy", [True, False], ids=["lazyframe", "dataframe"])
def test_prev_ranking_seeds_starting_ratings(monkeypatch, tmp_path, as_lazy):
	"""prev_ranking resumes an animal from its prior mu/sigma instead of the default.

	C never plays, so its emitted ordinal must equal the ordinal of the seeded
	rating throughout. Previously this branch raised (model undefined / wrong
	value type), so simply producing the right number is the regression guard.
	"""
	prev = pl.DataFrame(
		{"animal_id": ["C"], "mu": [40.0], "sigma": [2.0]},
		schema={"animal_id": pl.Utf8, "mu": pl.Float64, "sigma": pl.Float64},
	)
	prev_ranking = prev.lazy() if as_lazy else prev

	matches = [("B", "A", at(2023, 5, 24, 12, i, 0)) for i in range(3)]
	result = run_ranking(monkeypatch, matches, ["A", "B", "C"], prev_ranking, tmp_path)

	c_rows = result.filter(pl.col("animal_id") == "C")
	assert (c_rows["ordinal"] == expected_ordinal(mu=40.0, sigma=2.0)).all()


def test_use_prev_ranking_off_starts_fresh(monkeypatch, tmp_path):
	"""With use_prev_ranking off, a stored previous ranking is kept but not used."""
	prev = pl.DataFrame({"animal_id": ["C"], "mu": [40.0], "sigma": [2.0]})
	matches = [("B", "A", at(2023, 5, 24, 12, 0, 0))]

	result = run_ranking(
		monkeypatch, matches, ["A", "B", "C"], prev, tmp_path, use_prev_ranking=False
	)

	assert result.filter(animal_id="C")["ordinal"].item() == expected_ordinal()
	assert (tmp_path / Recording.PREV_RANKING).is_file()


def test_prev_ranking_defaults_animals_not_listed(monkeypatch, tmp_path):
	"""Animals absent from prev_ranking still start from the model default."""
	prev = pl.DataFrame(
		{"animal_id": ["C"], "mu": [40.0], "sigma": [2.0]},
		schema={"animal_id": pl.Utf8, "mu": pl.Float64, "sigma": pl.Float64},
	)
	# A and B are not in prev_ranking; with no matches between... use a B/A match
	# but check a third uninvolved default animal D.
	matches = [("B", "A", at(2023, 5, 24, 12, 0, 0))]
	result = run_ranking(monkeypatch, matches, ["A", "B", "C", "D"], prev, tmp_path)

	d_rows = result.filter(pl.col("animal_id") == "D")
	assert (d_rows["ordinal"] == expected_ordinal()).all()


# --- get_prev_ranking + round-trip ------------------------------------------


def test_get_prev_ranking_shape(monkeypatch):
	"""get_prev_ranking yields exactly the animal_id/mu/sigma prev_ranking shape."""
	matches = [("B", "A", at(2023, 5, 24, 12, i, 0)) for i in range(3)]
	result = run_ranking(monkeypatch, matches, ["A", "B"])

	prev = antenna_analysis.get_prev_ranking(result).collect()
	assert set(prev.columns) == {"animal_id", "mu", "sigma"}
	assert sorted(prev["animal_id"].to_list()) == ["A", "B"]
	assert prev.height == 2


def test_get_prev_ranking_takes_latest_rating(monkeypatch):
	"""The collapsed rating is the chronologically last one (matches final standings)."""
	matches = [("B", "A", at(2023, 5, 24, 12, i, 0)) for i in range(5)]
	result = run_ranking(monkeypatch, matches, ["A", "B"])

	prev = {
		r["animal_id"]: r
		for r in antenna_analysis.get_prev_ranking(result).collect().iter_rows(named=True)
	}
	for animal, ordinal in final_block(result).items():
		seeded = MODEL.rating(mu=prev[animal]["mu"], sigma=prev[animal]["sigma"])
		assert round(seeded.ordinal(), 3) == ordinal


def test_prev_ranking_from_a_full_trajectory_uses_the_chronologically_last_row(
	monkeypatch, tmp_path
):
	"""prev_ranking need not be pre-collapsed - set_prev_ranking runs it through
	get_prev_ranking, which sorts by datetime before taking each animal's last
	rating. An unsorted multi-row trajectory must still seed the chronologically LAST
	mu/sigma, not whatever row happens to land last in the frame.
	"""
	phase1 = [
		("B", "A", at(2023, 5, 24, 12, i, 0)) for i in range(5)
	]  # A's rating moves every match
	r1 = run_ranking(monkeypatch, phase1, ["A", "B", "C"])
	final1 = final_block(r1)

	shuffled = r1.sample(fraction=1.0, shuffle=True, seed=3)
	phase2 = [
		("B", "C", at(2023, 5, 25, 12, 0, 0))
	]  # A stays uninvolved, so its seed shows through
	result = run_ranking(monkeypatch, phase2, ["A", "B", "C"], shuffled, tmp_path)

	a_rows = result.filter(pl.col("animal_id") == "A")
	assert (a_rows["ordinal"] == final1["A"]).all()


def test_round_trip_continues_from_prev(monkeypatch, tmp_path):
	"""get_prev_ranking output fed back as prev_ranking carries ratings forward.

	Phase 1 lets A dominate B; phase 2 only has a C/D match, so A and B are
	uninvolved and must retain their phase-1 ratings via the seeded prev_ranking.
	"""
	phase1 = [("B", "A", at(2023, 5, 24, 12, i, 0)) for i in range(5)]
	r1 = run_ranking(monkeypatch, phase1, ["A", "B", "C", "D"])
	final1 = final_block(r1)

	prev = antenna_analysis.get_prev_ranking(r1)  # LazyFrame
	phase2 = [("D", "C", at(2023, 5, 25, 12, 0, 0))]
	r2 = run_ranking(monkeypatch, phase2, ["A", "B", "C", "D"], prev, tmp_path)
	final2 = final_block(r2)

	assert final2["A"] == final1["A"]
	assert final2["B"] == final1["B"]
	assert final2["A"] > final2["B"]


# --- empty match_df (no chasing events at all) ------------------------------


def test_empty_match_df_emits_empty_schema(monkeypatch):
	"""With no chasing events to replay, ranking returns an empty, well-typed frame.

	Exercises the dedicated ``else`` branch in calculate_ranking that builds an
	empty result from an explicit schema; the derived phase/day/hour columns are
	still appended, so the schema must match the populated case.
	"""
	result = run_ranking(monkeypatch, [], ["A", "B", "C"])

	assert result.height == 0
	assert set(result.columns) == {
		"animal_id",
		"mu",
		"sigma",
		"ordinal",
		"datetime",
		"phase",
		"day",
		"phase_count",
		"hour",
		"social_rank",
	}


# --- guards and determinism -------------------------------------------------


def test_prev_ranking_from_another_cohort_raises(monkeypatch, tmp_path):
	"""Seeding from a recording of different animals is a mistake, not a merge.

	Their ratings would be silently discarded while the current cohort started from
	scratch, so the continuation would look like it worked and mean nothing.
	"""
	prev = pl.DataFrame(
		{"animal_id": ["A", "Z"], "mu": [40.0, 30.0], "sigma": [2.0, 3.0]},
		schema={"animal_id": pl.Utf8, "mu": pl.Float64, "sigma": pl.Float64},
	)

	with pytest.raises(ValueError, match="not in the cohort"):
		run_ranking(
			monkeypatch, [("B", "A", at(2023, 5, 24, 12, 0, 0))], ["A", "B"], prev, tmp_path
		)


def test_get_prev_ranking_without_ratings_raises():
	"""A table without the rating columns is refused up front, naming what is missing."""
	with pytest.raises(ValueError, match="missing mu, sigma"):
		antenna_analysis.get_prev_ranking(pl.DataFrame({"animal_id": ["A"]}))


def test_tied_ordinals_share_the_same_rank_label(monkeypatch):
	"""Ranks are read off the ordinal, so animals on the same ordinal get the same label.

	A beats B and C beats D, each from the default rating, so the two winners end on one
	ordinal and the two losers on another. Naming a single dominant animal would mean
	picking arbitrarily between two that the data does not separate.
	"""
	matches = [
		("B", "A", at(2023, 5, 24, 12, 0, 0)),
		("D", "C", at(2023, 5, 24, 12, 1, 0)),
	]
	result = run_ranking(monkeypatch, matches, ["A", "B", "C", "D"])

	last = result.filter(pl.col("datetime") == result["datetime"].max())
	labels = dict(last.select("animal_id", "social_rank").iter_rows())

	assert labels == {
		"A": "dominant",
		"C": "dominant",
		"B": "subordinate",
		"D": "subordinate",
	}


def test_replay_matches_openskill_bit_for_bit(monkeypatch):
	"""The step's written-out one-on-one update gives exactly what ``model.rate`` gives.

	Long enough to reach the rare sigmas where ``sigma**2`` and ``sigma * sigma`` differ
	in the last bit, so the float operations have to match openskill's one for one.
	"""
	animals = list("ABCDEFGHI")
	rng = random.Random(1)
	pairs = [rng.sample(animals, 2) for _ in range(3000)]
	matches = [
		(loser, winner, at(2023, 5, 24, 12, 0, 0) + dt.timedelta(seconds=i))
		for i, (loser, winner) in enumerate(pairs)
	]
	result = run_ranking(monkeypatch, matches, animals)

	ratings = {animal: MODEL.rating() for animal in animals}
	expected = []
	for loser, winner in pairs:
		[[ratings[loser]], [ratings[winner]]] = MODEL.rate(
			[[ratings[loser]], [ratings[winner]]], ranks=[1, 0]
		)
		expected += [(ratings[a].mu, ratings[a].sigma) for a in animals]

	assert list(result.select("mu", "sigma").iter_rows()) == expected


def test_replay_is_deterministic_when_matches_share_a_timestamp(monkeypatch):
	"""Simultaneous events are replayed in one fixed order however they arrive.

	The ratings depend on the order the matches are applied in, and sorting on datetime
	alone leaves ties to the engine - so the same recording could publish different
	ranks on a re-run. The tiebreak is on the animals, which are unique per instant.
	"""
	moment = at(2023, 5, 24, 12, 0, 0)
	events = [("B", "A", moment), ("C", "B", moment), ("A", "C", moment)]

	forwards = run_ranking(monkeypatch, events, ["A", "B", "C"])
	backwards = run_ranking(monkeypatch, list(reversed(events)), ["A", "B", "C"])

	assert final_block(forwards) == final_block(backwards)
	# The tiebreak has to do something: the three animals do not end up level.
	assert len(set(final_block(forwards).values())) > 1
