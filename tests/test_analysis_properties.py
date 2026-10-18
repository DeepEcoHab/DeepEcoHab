"""Property/invariant tests for the antenna_analysis steps.

Where the per-function tests in ``test_*.py`` pin specific hand-computed values,
these draw randomised-but-valid inputs with Hypothesis and assert relationships
that must hold for *every* input -- cross-table consistency and the
missing-animal ("an animal dies / never shows up") contracts.

Inputs are kept inside a single light_phase occurrence (day 1, hour 12) so the
run-length ``phase_count`` of the hand-built frames matches the dense grid's
numbering (the phase_count footgun documented in strategies.py). Per animal the
generated cage stays never overlap in time, which is what the real data
guarantees and what the bitmask co-occupancy sweep relies on.

The steps read their input via ``Recording.load_results``; instead of the ``monkeypatch``
fixture (which trips Hypothesis's function-scoped-fixture health check) we swap the
attribute directly and restore it, and call the step directly.
"""

import datetime as dt
import math

import polars as pl
import strategies
from hypothesis import assume, given, settings, strategies as st

from deepecohab.core import antenna_analysis, transforms
from deepecohab.core.data_model import (
	CALENDAR_COLUMNS,
	AnalysisParams,
	DataFrameRegistry,
	Recording,
)

ANIMALS = ["A", "B", "C"]
RECORDING = strategies.analysis_recording(animal_ids=ANIMALS)
CAGES = strategies.ANALYSIS_CAGES
DIRECTIONAL = strategies.ANALYSIS_DIRECTIONAL
BASE = strategies.at(2023, 5, 24, 12, 0, 0)  # day-1 light_phase, hour 12


# --- input injection ---------------------------------------------------------
def run(fn, table_for_key, recording: Recording = RECORDING, **kwargs) -> pl.DataFrame:
	"""Run ``fn``'s pure body with ``Recording.load_results`` swapped for ``table_for_key``.

	``table_for_key`` is a ``(recording, key) -> LazyFrame`` callable, restored afterwards.
	``recording`` defaults to the linear fixture; the ring one is passed where the geometry
	matters.
	"""
	original = Recording.load_results
	Recording.load_results = lambda self, key, eager=False: table_for_key(self, key)
	try:
		return fn(recording, AnalysisParams(**kwargs)).collect()
	finally:
		Recording.load_results = original


# --- generators --------------------------------------------------------------
# One cage stay: (cage, gap-before-it, dwell). Bounded so a whole per-animal
# timeline (<= 4 stays) stays inside hour 12.
_stay = st.tuples(
	st.sampled_from(CAGES),
	st.floats(min_value=0, max_value=8, allow_nan=False),
	st.floats(min_value=0.5, max_value=20, allow_nan=False),
)
cage_plans = st.fixed_dictionaries({a: st.lists(_stay, max_size=4) for a in ANIMALS})


def padded_from_plan(plan: dict[str, list[tuple]]) -> pl.LazyFrame:
	"""Turn a per-animal stay plan into a padded_df.

	Each animal's stays are laid end-to-end (gap, dwell, gap, dwell, ...) from
	BASE, so one animal's intervals never overlap; different animals' do. The
	padded_df ``datetime`` marks the interval END.
	"""
	rows = []
	for animal, stays in plan.items():
		t = 0.0
		for cage, gap, dwell in stays:
			t += gap
			end = BASE + dt.timedelta(seconds=t + dwell)
			rows.append(
				{"animal_id": animal, "position": cage, "datetime": end, "time_spent": float(dwell)}
			)
			t += dwell
	return strategies.padded_df_frame(rows, RECORDING)


# winner/loser chasing events for match_df: distinct animals, a directional
# tunnel, all within hour 12.
_match_event = st.builds(
	lambda pair, tunnel, sec: {
		"winner": pair[0],
		"loser": pair[1],
		"position": tunnel,
		"datetime": BASE + dt.timedelta(seconds=float(sec)),
	},
	st.lists(st.sampled_from(ANIMALS), min_size=2, max_size=2, unique=True),
	st.sampled_from(DIRECTIONAL),
	st.floats(min_value=0, max_value=600, allow_nan=False),
)


# --- pairwise <-> activity consistency ---------------------------------------
@settings(max_examples=60, deadline=None)
@given(plan=cage_plans)
def test_pairwise_time_together_never_exceeds_either_occupancy(plan):
	"""Co-presence time of a pair in a cage can't exceed either animal's occupancy.

	Cross-table invariant between two steps reading the same padded_df: the time
	A and B are *together* in a cage is bounded by the time each of them spends
	there at all. minimum_time=0 keeps every meeting so the bound is tight.
	"""
	assume(any(plan.values()))  # the fully-empty padded_df is a separate extreme
	padded = padded_from_plan(plan)

	activity = run(antenna_analysis.calculate_activity, lambda recording, key: padded)
	pairwise = run(
		antenna_analysis.calculate_pairwise_meetings, lambda recording, key: padded, minimum_time=0
	)

	assert (pairwise["animal_id"].cast(pl.String) < pairwise["animal_id_2"].cast(pl.String)).all()

	occ = {
		(a, p): t
		for a, p, t in activity.group_by("animal_id", "position")
		.agg(pl.sum("time_in_position"))
		.iter_rows()
	}
	for r in pairwise.filter(pl.col("time_together") > dt.timedelta(0)).iter_rows(named=True):
		a, b, cage, tt = r["animal_id"], r["animal_id_2"], r["position"], r["time_together"]
		bound = min(occ.get((a, cage), dt.timedelta(0)), occ.get((b, cage), dt.timedelta(0)))
		assert tt <= bound + dt.timedelta(microseconds=1), (
			f"{a},{b}@{cage}: together {tt} > min-occupancy {bound}"
		)


# One cage stay that may span several cells: (cage, gap-before, dwell). Dwells reach
# past an hour mark and start on a fractional second, so padding really splits them --
# the exact scenario that used to inflate encounters.
_long_stay = st.tuples(
	st.sampled_from(CAGES),
	st.floats(min_value=0, max_value=1800, allow_nan=False),
	st.floats(min_value=0.5, max_value=7200, allow_nan=False),
)
long_cage_plans = st.fixed_dictionaries({a: st.lists(_long_stay, max_size=4) for a in ANIMALS})


def split_padded_from_plan(plan: dict[str, list[tuple]]) -> tuple[pl.LazyFrame, dict]:
	"""Lay each animal's stays end-to-end, then cut them at cells via real padding.

	Returns the padded_df and the per-(animal, cage) visit counts taken straight from
	the plan (one stay == one visit), so a test can bound encounters by visit totals.
	"""
	rows = []
	visits: dict[tuple[str, str], int] = {}
	for animal, stays in plan.items():
		t = 0.5  # fractional second so pieces have fractional time_spent
		for cage, gap, dwell in stays:
			t += gap
			end = BASE + dt.timedelta(seconds=t + dwell)
			rows.append(
				{"animal_id": animal, "position": cage, "datetime": end, "time_spent": float(dwell)}
			)
			visits[(animal, cage)] = visits.get((animal, cage), 0) + 1
			t += dwell
	frame = pl.DataFrame(
		{
			"animal_id": pl.Series([r["animal_id"] for r in rows], dtype=pl.Enum(ANIMALS)),
			"position": pl.Series([r["position"] for r in rows], dtype=pl.Categorical),
			"datetime": pl.Series("datetime", [r["datetime"] for r in rows]),
			"time_spent": strategies.seconds([r["time_spent"] for r in rows]),
			"time_under": pl.Series([dt.timedelta(0) for _ in rows], dtype=pl.Duration("us")),
		}
	)
	return transforms.split_visits_on_cells(frame.lazy(), RECORDING), visits


@settings(max_examples=60, deadline=None)
@given(plan=long_cage_plans)
def test_pairwise_encounters_never_exceed_visit_sum(plan):
	"""A pair can't meet in a cage more often than their combined visits to it.

	The user's invariant. Stays span hour marks so padding splits them into many
	pieces; the step must merge those pieces back per visit rather than count each
	piece as a fresh encounter (the bug that pushed encounters above the visit sum).
	minimum_time=0 keeps every meeting, making the bound tightest.
	"""
	assume(any(plan.values()))
	padded, visits = split_padded_from_plan(plan)
	pairwise = run(
		antenna_analysis.calculate_pairwise_meetings, lambda recording, key: padded, minimum_time=0
	)

	for r in pairwise.filter(pl.col("pairwise_encounters") > 0).iter_rows(named=True):
		a, b, cage, enc = r["animal_id"], r["animal_id_2"], r["position"], r["pairwise_encounters"]
		bound = visits.get((a, cage), 0) + visits.get((b, cage), 0)
		assert enc <= bound, f"{a},{b}@{cage}: {enc} encounters > {bound} combined visits"


# --- chasings totals reconcile with the event table -------------------------
@settings(max_examples=60, deadline=None)
@given(events=st.lists(_match_event, min_size=0, max_size=12))
def test_chasings_grid_total_equals_event_count(events):
	"""Summing the per-hour chasings recovers exactly the number of events.

	calculate_chasings only aggregates, so no event may be lost or invented.
	"""
	match_df = strategies.match_df_frame(events, RECORDING)
	chasings = run(antenna_analysis.calculate_chasings, lambda recording, key: match_df)

	assert int(chasings["chasings"].sum()) == len(events)

	# Per-chaser totals reconcile with the raw winners.
	expected: dict[str, int] = {}
	for e in events:
		expected[e["winner"]] = expected.get(e["winner"], 0) + 1
	got = {
		a: int(n)
		for a, n in chasings.group_by("chaser")
		.agg(pl.sum("chasings"))
		.filter(pl.col("chasings") > 0)
		.iter_rows()
	}
	assert got == expected


# --- missing-animal contracts ------------------------------------------------
@settings(max_examples=40, deadline=None)
@given(events=st.lists(_match_event, min_size=1, max_size=12), absent=st.sampled_from(ANIMALS))
def test_absent_animal_ranking_stays_frozen(events, absent):
	"""An animal that is in no match keeps its starting rating for the whole run.

	The 'animal dies / never interacts' case: ranking emits a row per animal after
	every match, and an animal never named as winner or loser must show a single,
	unchanged mu/sigma across all of them (never dropped, never drifting).
	"""
	events = [e for e in events if absent not in (e["winner"], e["loser"])]
	assume(events)  # need at least one match to drive the ranking

	match_df = strategies.match_df_frame(events, RECORDING)
	ranking = run(
		antenna_analysis.calculate_ranking, lambda recording, key: match_df, use_prev_ranking=False
	)

	rows = ranking.filter(pl.col("animal_id") == absent)
	assert rows.height == len(events)  # present in the trajectory after every match
	assert rows["mu"].n_unique() == 1
	assert rows["sigma"].n_unique() == 1
	assert rows["ordinal"].unique().to_list() == [0.0]  # default rating's ordinal


@settings(max_examples=40, deadline=None)
@given(plan=cage_plans, absent=st.sampled_from(ANIMALS))
def test_absent_animal_zero_filled_in_activity(plan, absent):
	"""An animal absent from padded_df still occupies every grid cell, all-zero.

	Locks the dense-grid contract for activity: a missing animal is present with
	zero occupancy / visits / solitary time, never silently dropped from the table.
	"""
	plan = {a: ([] if a == absent else stays) for a, stays in plan.items()}
	assume(any(plan.values()))
	padded = padded_from_plan(plan)

	activity = run(antenna_analysis.calculate_activity, lambda recording, key: padded)

	rows = activity.filter(pl.col("animal_id") == absent)
	assert rows.height > 0
	assert rows["time_in_position"].sum() == dt.timedelta(0)
	assert rows["visits_to_position"].sum() == 0
	assert rows["time_alone"].sum() == dt.timedelta(0)
	# Every position still represented for the absent animal.
	assert set(rows["position"].cast(pl.String).unique()) == set(
		RECORDING.layout.positions_non_directional
	)


# --- features stay finite for an animal with no interactions -----------------
def test_absent_animal_appears_in_features_with_finite_values(monkeypatch):
	"""A behaviourally-silent animal surfaces in features via the dense activity grid.

	C never chases, is never chased and shares no cage time, but activity_df is dense
	(C present with zeros), so features must still list C for every metric with
	finite, non-null value and exposure.
	"""
	PHASE = pl.Enum(["light_phase", "dark_phase"])
	AN = pl.Enum(ANIMALS)

	def base(n):
		return {
			"phase": pl.Series(["light_phase"] * n, dtype=PHASE),
			"day": pl.Series([1] * n, dtype=pl.UInt16),
			"phase_count": pl.Series([1] * n, dtype=pl.UInt16),
			"hour": pl.Series([12] * n, dtype=pl.UInt8),
		}

	chasings = pl.LazyFrame(
		{
			**base(1),
			"chaser": pl.Series(["A"], dtype=AN),
			"chased": pl.Series(["B"], dtype=AN),
			"chasings": pl.Series([1], dtype=pl.UInt32),
		}
	)
	# activity is dense: all three animals present, C with zeros.
	activity = pl.LazyFrame(
		{
			**base(3),
			"position": pl.Series(["cage_1"] * 3, dtype=pl.Categorical),
			"animal_id": pl.Series(["A", "B", "C"], dtype=AN),
			"visits_to_position": pl.Series([10, 20, 0], dtype=pl.UInt32),
			"time_alone": strategies.seconds([1.0, 2.0, 0.0]),
			"time_in_position": strategies.seconds([3600.0, 3600.0, 0.0]),
		}
	)
	pairwise = pl.LazyFrame(
		{
			**base(1),
			# A cage only because the row needs a position; features sums over all of them.
			"position": pl.Series(["cage_1"], dtype=pl.Categorical),
			"animal_id": pl.Series(["A"], dtype=AN),
			"animal_id_2": pl.Series(["B"], dtype=AN),
			"time_together": strategies.seconds([5.0]),
			"pairwise_encounters": pl.Series([1], dtype=pl.UInt32),
		}
	)
	# One row per antenna registration: A seen 5 times, B 3, C never.
	main = pl.LazyFrame({**base(8), "animal_id": pl.Series(["A"] * 5 + ["B"] * 3, dtype=AN)})
	tables = {
		"chasings_df": chasings,
		"pairwise_meetings": pairwise,
		"activity_df": activity,
		"main_df": main,
	}
	monkeypatch.setattr(Recording, "load_results", lambda self, key, eager=False: tables[key])

	result = antenna_analysis.calculate_features(RECORDING, AnalysisParams()).collect()

	c_rows = result.filter(pl.col("animal_id") == "C")
	assert set(c_rows["metric"].unique()) == {
		"time_alone",
		"n_chasing",
		"n_chased",
		"activity",
		"time_together",
		"pairwise_encounters",
		"n_chasing_per_detection",
	}
	for column in ("value", "exposure"):
		assert c_rows[column].null_count() == 0
		assert all(math.isfinite(v) for v in c_rows[column].to_list())

	# A chased once over 5 detections; the rate is sum(value) / sum(exposure).
	per_detection = result.filter(
		pl.col("animal_id") == "A", pl.col("metric") == "n_chasing_per_detection"
	)
	assert per_detection["value"].sum() == 1.0
	assert per_detection["exposure"].sum() == 5.0


# --- the tiling invariant ----------------------------------------------------
# A whole-pipeline case: raw reads -> main_df -> padded_df -> activity_df. Everything
# above starts from a hand-built padded_df, so only this one can say whether the real
# padding actually tiles an animal's timeline.
TILED = strategies.analysis_recording(
	animal_ids=ANIMALS, start="2023-05-24 00:00:00", finish="2023-05-25 00:00:00"
)
# One raw registration: which animal, how many seconds into the window, which antenna.
_read = st.tuples(
	st.sampled_from(ANIMALS),
	st.integers(min_value=0, max_value=24 * 3600),
	st.sampled_from([1, 2, 3, 4]),
)


def pipeline_from_reads(reads: list[tuple], target: str) -> dict[str, pl.DataFrame]:
	"""Every real step up to ``target`` over raw registrations, keyed by step name."""
	start, _ = TILED.timeline.local_span
	TILED.data = pl.DataFrame(
		[
			{
				"datetime": start + dt.timedelta(seconds=offset),
				"antenna": antenna,
				"time_under": dt.timedelta(milliseconds=100),
				"animal_id": animal,
				"inserted": False,
			}
			for animal, offset, antenna in reads
		],
		schema=TILED.data_schema,
	).lazy()

	tables: dict[str, pl.DataFrame] = {}
	original = Recording.load_results
	Recording.load_results = lambda self, key, eager=False: tables[key].lazy()
	try:
		for step in DataFrameRegistry.step_order([target]):
			tables[step] = DataFrameRegistry._builders[step](TILED, AnalysisParams()).collect()
	finally:
		Recording.load_results = original
	return tables


@settings(max_examples=30, deadline=None)
@given(reads=st.lists(_read, min_size=1, max_size=10))
def test_every_hour_inside_an_animal_s_observed_span_is_fully_tiled(reads):
	"""An animal is somewhere at every instant, so its hour sums to exactly 3600 s.

	This is what makes a rate in feature_df mean anything: ``observed_hours`` is read as
	the time the animal was watched for, and every metric is divided by it. Gaps would
	inflate every rate, and double-counted pieces would deflate them. Positions include
	``undefined``, which is the point - unplaceable time still has to be time.

	Exact, to the microsecond: the pieces are cut from the same integer-microsecond
	bounds they are summed back over, so there is no rounding to tolerate.
	"""
	start, end = TILED.timeline.local_span
	first_read = {}
	for animal, offset, _ in reads:
		moment = start + dt.timedelta(seconds=offset)
		first_read[animal] = min(first_read.get(animal, moment), moment)

	totals = (
		pipeline_from_reads(reads, "activity_df")["activity_df"]
		.group_by("animal_id", "day", "hour")
		.agg(pl.sum("time_in_position"))
	)

	for row in totals.iter_rows(named=True):
		bin_start = start + dt.timedelta(days=row["day"] - 1, hours=row["hour"])
		bin_end = bin_start + dt.timedelta(hours=1)
		if row["animal_id"] not in first_read:
			continue  # an animal never registered is zero-filled, not tiled
		if bin_start < first_read[row["animal_id"]] or bin_end > end:
			continue  # only hours the animal was observed for the whole of
		assert row["time_in_position"] == dt.timedelta(hours=1), row


# --- per-position features ---------------------------------------------------
# Reads packed into a few hours, so animals share cages often enough to meet.
_packed_read = st.tuples(
	st.sampled_from(ANIMALS),
	st.integers(min_value=0, max_value=4 * 3600),
	st.sampled_from([1, 2, 3, 4]),
)
PLACED = ["activity", "time_alone", "time_together", "pairwise_encounters"]
PAIRED_PLACED = {"time_together", "pairwise_encounters"}


@settings(max_examples=40, deadline=None)
@given(reads=st.lists(_packed_read, min_size=2, max_size=30))
def test_features_per_position_are_each_position_s_share_of_the_whole(reads):
	"""Each position's row is that position's own count, exposed against the time there.

	Checked against activity_df and pairwise_meetings directly: every position of the
	dense grid has exactly one row per placed metric, carrying the upstream value, and
	its exposure is the time spent there - per partner for a paired metric. Summed over
	positions that exposure is the animal's whole observed time, the denominator every
	rate had before positions were kept. Within one position a fraction of time cannot
	pass 1: nobody is alone, or with a partner, for longer than they were there.
	"""
	tables = pipeline_from_reads(reads, "feature_df")
	features = tables["feature_df"]
	partners = TILED.cohort.n_mice - 1
	keys = [*CALENDAR_COLUMNS, "animal_id", "position"]

	def hours(column: str) -> pl.Expr:
		return pl.col(column).dt.total_hours(fractional=True)

	together = (
		tables["pairwise_meetings"]
		.unpivot(
			on=["animal_id", "animal_id_2"],
			index=[*CALENDAR_COLUMNS, "position", "time_together", "pairwise_encounters"],
			value_name="animal",
		)
		.drop("variable")
		.rename({"animal": "animal_id"})
		.group_by(keys)
		.agg(hours("time_together").sum(), pl.sum("pairwise_encounters").cast(pl.Float64))
	)
	expected = (
		tables["activity_df"]
		.select(
			*keys,
			pl.col("visits_to_position").cast(pl.Float64).alias("activity"),
			hours("time_alone"),
			hours("time_in_position").alias("hours_there"),
		)
		.join(together, on=keys, how="left")
		.with_columns(pl.col("time_together", "pairwise_encounters").fill_null(0.0))
		.unpivot(on=PLACED, index=[*keys, "hours_there"], variable_name="metric")
		.with_columns(
			pl.when(pl.col("metric").is_in(PAIRED_PLACED))
			.then(pl.col("hours_there") * partners)
			.otherwise(pl.col("hours_there"))
			.alias("expected_exposure")
		)
	)

	placed = features.filter(pl.col("position").is_not_null())
	joined = placed.join(expected, on=[*keys, "metric"], how="full", coalesce=True)

	assert placed.height == expected.height == joined.height
	assert set(placed["metric"]) == set(PLACED)
	for got, want in (("value", "value_right"), ("exposure", "expected_exposure")):
		assert (joined[got] - joined[want]).abs().max() < 1e-12

	fractions = placed.filter(pl.col("metric").is_in(["time_alone", "time_together"]))
	assert (fractions["value"] <= fractions["exposure"] + 1e-12).all()

	observed = (
		tables["activity_df"]
		.group_by(*CALENDAR_COLUMNS, "animal_id")
		.agg(hours("time_in_position").sum().alias("observed"))
	)
	summed = (
		placed.group_by(*CALENDAR_COLUMNS, "animal_id", "metric")
		.agg(pl.sum("exposure"))
		.join(observed, on=[*CALENDAR_COLUMNS, "animal_id"])
		.with_columns(
			pl.when(pl.col("metric").is_in(PAIRED_PLACED))
			.then(pl.col("observed") * partners)
			.otherwise(pl.col("observed"))
		)
	)
	assert (summed["exposure"] - summed["observed"]).abs().max() < 1e-12

	unplaced = features.filter(pl.col("position").is_null())
	assert set(unplaced["metric"]) == {"n_chasing", "n_chased", "n_chasing_per_detection"}
	assert unplaced.select(*CALENDAR_COLUMNS, "animal_id", "metric").is_duplicated().sum() == 0


# --- match_df against an independent oracle ----------------------------------
# The ring layout, where each cage sits between two antennas, so an animal walking the
# ring resolves alternately to a cage and to a tunnel. The linear fixture above has one
# antenna per cage, where only a repeat read places an animal in a cage - so a chase,
# which needs the winner seen leaving a cage, almost never arises there.
RING = strategies.ring_recording(animal_ids=ANIMALS)
RING_TUNNELS = RING.layout.tunnel_names_directional

# One step: who moves, which way around the antenna ring, how long after the previous read
# anywhere in the habitat, and how far behind each follower comes. A step may move several
# animals through the same antenna in quick succession, which is what a chase looks like in
# the raw reads; the gaps straddle the chasing window, so accepted and rejected
# follow-throughs are both generated.
_antenna_step = st.tuples(
	st.lists(st.sampled_from(ANIMALS), min_size=1, max_size=len(ANIMALS), unique=True),
	st.sampled_from([-1, 0, 1, 1]),
	st.sampled_from([0.05, 0.2, 0.5, 0.9, 1.5]),
	st.sampled_from([0.2, 0.5, 0.9, 1.5]),
)


def main_df_rows_from_steps(steps: list[tuple]) -> list[dict]:
	"""Walk the animals around the ring and resolve the main_df rows that produces."""
	ring = len(RING.layout.cages) * 2
	clock, antenna = 0.0, dict.fromkeys(ANIMALS, 1)
	walks: dict[str, list[tuple]] = {animal: [] for animal in ANIMALS}

	for movers, move, delta, follow_gap in steps:
		clock += delta  # every read gets its own instant, so no tie can reorder a walk
		for index, animal in enumerate(movers):
			clock += follow_gap if index else 0.0
			antenna[animal] = (antenna[animal] - 1 + move) % ring + 1
			walks[animal].append((antenna[animal], BASE + dt.timedelta(seconds=clock)))

	return [
		row
		for animal, walk in walks.items()
		for row in strategies.antenna_reads(animal, walk, RING)
	]


@settings(max_examples=60, deadline=None)
@given(steps=st.lists(_antenna_step, min_size=2, max_size=25))
def test_matches_agree_with_the_brute_force_oracle(steps):
	"""The sweep, the bitmask decode and the as-of join find exactly what a loop does.

	Three animals shuffling between neighbouring cages produce repeat passes, head-on
	crossings, several animals inside one tunnel and exits just outside the window - the
	cases the hand-written tests each isolate, here all at once and compared against a
	reference that shares none of the machinery.
	"""
	rows = main_df_rows_from_steps(steps)
	main = strategies.main_df_frame(rows, RING)

	result = run(antenna_analysis.calculate_matches, lambda recording, key: main, recording=RING)

	produced = sorted(
		(row["position"], row["winner"], row["loser"], row["datetime"], row["chasing_length"])
		for row in result.iter_rows(named=True)
	)
	assert produced == strategies.expected_matches(rows, RING, AnalysisParams().chasing_time_window)
