"""Tests for calculate_incohort_sociability (observed togetherness minus chance).

For each pair and cage, observed co-presence (as a fraction of phase duration) is
compared to the chance expectation from each animal's independent occupancy
(product of occupancy proportions); sociability is the observed-minus-chance
difference summed over cages (DOI:10.7554/eLife.19532).

The step reads three upstream tables via Recording.load_results — ``activity_df``,
``pairwise_meetings`` and ``phase_durations`` — so we monkeypatch it with a
key-dispatching stub and call the step directly. All fixtures live in a
single light_phase occurrence (day 1, phase_count 1) so the arithmetic is by hand.
"""

import polars as pl
import pytest
import strategies

from deepecohab.core import antenna_analysis
from deepecohab.core.data_model import AnalysisParams, Recording

RECORDING = strategies.analysis_recording(animal_ids=["A", "B", "C"])
PHASE_ENUM = pl.Enum(["light_phase", "dark_phase"])
ANIMAL_ENUM = pl.Enum(RECORDING.cohort.animal_tags)


def activity_frame(
	entries: list[tuple[str, str, float]], phase: str = "light_phase", phase_count: int = 1
) -> pl.LazyFrame:
	"""activity_df rows (animal_id, position, time_in_position) in one phase of day 1."""
	n = len(entries)
	return pl.LazyFrame(
		{
			"phase": pl.Series([phase] * n, dtype=PHASE_ENUM),
			"day": pl.Series([1] * n, dtype=pl.UInt16),
			"phase_count": pl.Series([phase_count] * n, dtype=pl.UInt16),
			"hour": pl.Series([12] * n, dtype=pl.UInt8),
			"position": pl.Series([e[1] for e in entries], dtype=pl.Categorical),
			"animal_id": pl.Series([e[0] for e in entries], dtype=ANIMAL_ENUM),
			"time_in_position": strategies.seconds([float(e[2]) for e in entries]),
		}
	)


def pairwise_frame(
	entries: list[tuple[str, str, str, float]], phase: str = "light_phase", phase_count: int = 1
) -> pl.LazyFrame:
	"""pairwise_meetings rows (a, b, position, time_together) in one phase of day 1."""
	n = len(entries)
	return pl.LazyFrame(
		{
			"phase": pl.Series([phase] * n, dtype=PHASE_ENUM),
			"day": pl.Series([1] * n, dtype=pl.UInt16),
			"phase_count": pl.Series([phase_count] * n, dtype=pl.UInt16),
			"hour": pl.Series([12] * n, dtype=pl.UInt8),
			"position": pl.Series([e[2] for e in entries], dtype=pl.Categorical),
			"animal_id": pl.Series([e[0] for e in entries], dtype=ANIMAL_ENUM),
			"animal_id_2": pl.Series([e[1] for e in entries], dtype=ANIMAL_ENUM),
			"time_together": strategies.seconds([float(e[3]) for e in entries]),
			"pairwise_encounters": pl.Series([1] * n, dtype=pl.UInt32),
		}
	)


def durations_frame(
	duration_seconds: float, phase: str = "light_phase", phase_count: int = 1
) -> pl.LazyFrame:
	"""One phase occurrence's wall-clock length; concat several for a multi-phase case."""
	return pl.LazyFrame(
		{
			"phase": pl.Series([phase], dtype=PHASE_ENUM),
			"phase_count": pl.Series([phase_count], dtype=pl.UInt16),
			"duration": strategies.seconds([float(duration_seconds)]),
		}
	)


def run_sociability(monkeypatch, *, activity, pairwise, durations) -> pl.DataFrame:
	tables = {
		"activity_df": activity,
		"pairwise_meetings": pairwise,
		"phase_durations": durations,
	}
	monkeypatch.setattr(Recording, "load_results", lambda self, key, eager=False: tables[key])
	return antenna_analysis.calculate_incohort_sociability(RECORDING, AnalysisParams()).collect()


def pair_row(result: pl.DataFrame, a: str, b: str) -> dict:
	return result.filter((pl.col("animal_id") == a) & (pl.col("animal_id_2") == b)).row(
		0, named=True
	)


def test_tunnel_time_together_is_excluded(monkeypatch):
	"""Tunnel co-presence never reaches sociability, so the metric is unchanged by it.

	pairwise_meetings covers tunnels, but the chance term is built from cage occupancy
	only; an unfiltered tunnel row would survive the left join with a null chance and
	inflate proportion_together while leaving sociability alone.
	"""
	# A in cage_1 for 40 s, B for 30 s, together 20 s there and 50 s in a tunnel.
	activity = activity_frame([("A", "cage_1", 40), ("B", "cage_1", 30)])
	pairwise = pairwise_frame([("A", "B", "cage_1", 20), ("A", "B", "tunnel_1", 50)])

	result = run_sociability(
		monkeypatch, activity=activity, pairwise=pairwise, durations=durations_frame(100)
	)
	row = pair_row(result, "A", "B")

	assert row["proportion_together"] == pytest.approx(0.2)  # 20 / 100, tunnel ignored
	assert row["sociability"] == pytest.approx(0.2 - (40 * 30) / 100**2)


def test_sociability_sums_over_cages(monkeypatch):
	"""Sociability is summed across cages."""
	activity = activity_frame(
		[("A", "cage_1", 40), ("B", "cage_1", 30), ("A", "cage_2", 10), ("B", "cage_2", 10)]
	)
	pairwise = pairwise_frame([("A", "B", "cage_1", 20), ("A", "B", "cage_2", 5)])
	result = run_sociability(
		monkeypatch, activity=activity, pairwise=pairwise, durations=durations_frame(100)
	)

	cage1 = 20 / 100 - (40 * 30) / 100**2
	cage2 = 5 / 100 - (10 * 10) / 100**2
	row = pair_row(result, "A", "B")
	assert row["sociability"] == pytest.approx(cage1 + cage2)
	assert row["proportion_together"] == pytest.approx(20 / 100 + 5 / 100)


def test_each_phase_occurrence_is_reported_on_its_own(monkeypatch):
	"""The metric is per phase, so two occurrences never pool into one number.

	The same pair is sociable in the light phase and avoidant in the dark one; a single
	row averaging the two would report neither.
	"""
	activity = pl.concat(
		[
			activity_frame([("A", "cage_1", 40), ("B", "cage_1", 30)]),
			activity_frame([("A", "cage_1", 40), ("B", "cage_1", 30)], "dark_phase", 2),
		]
	)
	pairwise = pl.concat(
		[
			pairwise_frame([("A", "B", "cage_1", 30)]),
			pairwise_frame([("A", "B", "cage_1", 5)], "dark_phase", 2),
		]
	)
	durations = pl.concat([durations_frame(100), durations_frame(100, "dark_phase", 2)])

	result = run_sociability(monkeypatch, activity=activity, pairwise=pairwise, durations=durations)

	by_phase = dict(result.select("phase_count", "sociability").iter_rows())
	assert by_phase[1] == pytest.approx(0.3 - (40 * 30) / 100**2)
	assert by_phase[2] == pytest.approx(0.05 - (40 * 30) / 100**2)
	assert by_phase[1] > 0 > by_phase[2]


def test_the_duration_is_looked_up_per_phase_occurrence(monkeypatch):
	"""Each occurrence divides by its own length, not by the first one's.

	A recording cut short leaves its last phase shorter than the rest, so joining the
	wrong duration would rescale every metric in it.
	"""
	activity = pl.concat(
		[
			activity_frame([("A", "cage_1", 40), ("B", "cage_1", 30)]),
			activity_frame([("A", "cage_1", 40), ("B", "cage_1", 30)], "dark_phase", 2),
		]
	)
	pairwise = pl.concat(
		[
			pairwise_frame([("A", "B", "cage_1", 20)]),
			pairwise_frame([("A", "B", "cage_1", 20)], "dark_phase", 2),
		]
	)
	# The dark phase is half as long, so the same 20 s is twice the proportion.
	durations = pl.concat([durations_frame(100), durations_frame(50, "dark_phase", 2)])

	result = run_sociability(monkeypatch, activity=activity, pairwise=pairwise, durations=durations)

	proportions = dict(result.select("phase_count", "proportion_together").iter_rows())
	assert proportions[1] == pytest.approx(0.2)
	assert proportions[2] == pytest.approx(0.4)


def test_a_pair_never_together_sits_at_minus_the_chance_sum(monkeypatch):
	"""Avoidance is bounded below by how often chance alone would have met them.

	Both animals use both cages but never at the same time, so the observed term is zero
	in every cage and the metric is exactly the negated chance expectation - the floor
	the measure can reach for that pair. pairwise_meetings stores no row for a cage a pair
	never met in, so the chance terms must come through without one.
	"""
	activity = activity_frame(
		[("A", "cage_1", 40), ("B", "cage_1", 30), ("A", "cage_2", 20), ("B", "cage_2", 10)]
	)
	pairwise = pairwise_frame([])

	result = run_sociability(
		monkeypatch, activity=activity, pairwise=pairwise, durations=durations_frame(100)
	)
	row = pair_row(result, "A", "B")

	chance = (40 * 30) / 100**2 + (20 * 10) / 100**2
	assert row["proportion_together"] == pytest.approx(0.0)
	assert row["sociability"] == pytest.approx(-chance)
