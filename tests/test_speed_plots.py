"""Unit tests for speed preparation; dashboard behavior is checked in the live app."""

import datetime as dt

import polars as pl
import pytest

from deepecohab.plotting import prepare
from deepecohab.plotting.animals import mean_by_group, resolve_colors
from deepecohab.plotting.context import PlotContext


@pytest.fixture
def context():
	return PlotContext(
		animal_ids=["a", "b"],
		cages=["cage"],
		positions=["cage", "t1", "t2", "undefined"],
		phases={"light_phase": 0, "dark_phase": 12},
		days_range=(1, 3),
		phase_range=(1, 6),
		tunnels_map={"forward": "t1", "reverse": "t1", "other": "t2"},
		tunnel_lengths_cm={"t1": 30, "t2": 60},
		_loaded={"animals": pl.DataFrame({"animal_id": ["a", "b"]})},
	)


def crossings(context, durations, **columns):
	n = len(durations)
	days = columns.get("day", [1] * n)
	context._loaded["main_df"] = pl.DataFrame(
		{
			"animal_id": ["a"] * n,
			"position": ["forward"] * n,
			"time_spent": [dt.timedelta(seconds=s) if s is not None else None for s in durations],
			"day": days,
			"hour": [2] * n,
			"phase_count": [2 * day - 1 for day in days],
			"phase": ["light_phase"] * n,
			**columns,
		}
	)


def test_distance_and_fractional_duration_in_both_directions(context):
	crossings(context, [0.5, 2, 2], position=["forward", "reverse", "other"])
	frame = prepare.prep_animal_speed(context, (1, 3), ["light_phase"])
	assert frame.select("position", "speed_cm_s").rows() == [("t1", 15), ("t2", 30), ("t1", 60)]


def test_cutoff_is_inclusive_and_excludes_invalid_crossings(context):
	crossings(
		context, [0, -1, None, 10, 10.1, 1, 1], position=["forward"] * 5 + ["cage", "undefined"]
	)
	frame = prepare.prep_animal_speed(context, (1, 3), ["light_phase"])
	assert frame["speed_cm_s"].to_list() == [3]


def test_cutoff_can_be_changed(context):
	crossings(context, [0.5, 2, 12])
	frame = prepare.prep_animal_speed(context, (1, 3), ["light_phase"], max_dwell=12)
	assert frame["speed_cm_s"].to_list() == [2.5, 15, 60]


@pytest.mark.parametrize("cutoff", [0, -1, float("nan"), float("inf")])
def test_invalid_cutoff_is_rejected(context, cutoff):
	with pytest.raises(ValueError, match="max_dwell"):
		prepare.prep_animal_speed(context, (1, 3), ["light_phase"], max_dwell=cutoff)


@pytest.mark.parametrize("length", [None, 0, -1, float("nan"), float("inf")])
def test_unknown_or_invalid_distance_never_implies_twenty_cm(context, length):
	if length is None:
		del context.tunnel_lengths_cm["t1"]
	else:
		context.tunnel_lengths_cm["t1"] = length
	with pytest.raises(ValueError, match="crossing length"):
		prepare.prep_animal_speed(context, (1, 3), ["light_phase"])


def test_phase_window_phase_type_and_hours_filter_before_aggregation(context):
	crossings(
		context,
		[1, 2, 3, 4],
		day=[1, 2, 2, 3],
		hour=[2, 2, 3, 2],
		phase=["dark_phase", "light_phase", "light_phase", "light_phase"],
	)
	frame = prepare.prep_animal_speed(context, (3, 3), ["light_phase"], "phase_count", (2, 2))
	assert frame["speed_cm_s"].to_list() == [15]


@pytest.mark.parametrize("granularity, units", [("day", [1, 2, 1]), ("phase_count", [1, 3, 1])])
def test_boxes_pool_one_median_per_animal_tunnel_and_unit(context, granularity, units):
	crossings(
		context,
		[1, 2, 3, 2, 1],
		position=["forward", "reverse", "forward", "forward", "other"],
		day=[1, 1, 1, 2, 1],
		animal_id=["a", "a", "a", "a", "b"],
	)
	frame = prepare.prep_speed_box(context, (1, 3), ["light_phase"], granularity)
	assert frame.select("position", "animal_id", "speed_cm_s").rows() == [
		("t1", "a", 15),
		("t1", "a", 15),
		("t2", "b", 60),
	]
	assert frame[granularity].to_list() == units


def test_hourly_mean_weights_observed_days_equally_and_has_sem(context):
	crossings(context, [1, 1, 3], day=[1, 1, 2])
	frame = prepare.prep_speed_line(context, (1, 3), ["light_phase"], "day", "hour")
	row = frame.filter((pl.col("animal_id") == "a") & (pl.col("hour") == 2)).row(0, named=True)
	assert row["mean"] == 20  # Cell means 30 and 10; not a pooled crossing mean of 23.33.
	assert row["sem"] == pytest.approx(10)
	assert row["lower"] == pytest.approx(10)
	assert row["upper"] == pytest.approx(30)
	assert frame.filter(pl.col("mean").is_not_null()).height == 1


def test_daily_mean_weights_observed_hours_equally(context):
	crossings(context, [1, 1, 3], hour=[2, 2, 3])
	frame = prepare.prep_speed_line(context, (1, 3), ["light_phase"], "day", "day")
	row = frame.filter((pl.col("animal_id") == "a") & (pl.col("day") == 1)).row(0, named=True)
	assert row["mean"] == 20
	assert row["sem"] == pytest.approx(10)


def test_phase_axis_keeps_empty_bins_null_and_single_sample_sem_unknown(context):
	crossings(context, [1, 3], day=[1, 2])
	frame = prepare.prep_speed_line(context, (1, 3), ["light_phase"], "phase_count", "phase_count")
	a = frame.filter(pl.col("animal_id") == "a")
	assert a["mean"].to_list() == [30, None, 10]
	assert a["sem"].to_list() == [None, None, None]


def test_empty_selection_does_not_produce_zero_speed(context):
	crossings(context, [1])
	assert prepare.prep_speed_box(context, (1, 3), ["dark_phase"], "day").is_empty()
	frame = prepare.prep_speed_line(context, (1, 3), ["dark_phase"], "day", "hour", (2, 3))
	assert frame.height == 4
	assert frame["mean"].null_count() == 4


def test_group_mean_ignores_animals_without_crossings(context):
	crossings(context, [2])
	context._loaded["animals"] = pl.DataFrame({"animal_id": ["a", "b"], "sex": ["M", "M"]})
	frame = prepare.prep_speed_line(context, (1, 3), ["light_phase"], "day", "hour")
	grouped = mean_by_group(frame, resolve_colors(context, "sex", group_mean=True), ["mean"])
	assert grouped["mean"].drop_nulls().to_list() == [15]
