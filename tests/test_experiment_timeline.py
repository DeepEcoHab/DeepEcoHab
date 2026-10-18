"""Tests for the egocentric experiment timeline.

Recordings are compared by where they are in their own experiment rather than by wall
clock, so day 1 hour 0 is the first onset of the phase named in ``start_from`` and
anything recorded before it is trimmed; the experiment ends at the close of the
phase named in ``end_with``. These tests cover resolving that origin and the
grid that follows from it - including the case the minute walk exists for, where one
hourly bin holds two phases.
"""

import datetime as dt
from functools import partial
from zoneinfo import ZoneInfo

import polars as pl
import pytest
import strategies
from pydantic import ValidationError

from deepecohab.core import grids
from deepecohab.core.data_model import AnalysisParams, Recording, Timeline
from deepecohab.core.recording_pipeline import build_main_df

TZ_NAME = "Europe/Warsaw"
TZ = ZoneInfo(TZ_NAME)

# Onsets a whole hour apart, as a facility timer is normally set.
ALIGNED = {"light_phase": dt.time(7, 0), "dark_phase": dt.time(20, 0)}
# Onsets at different minute offsets, so bins anchored on one straddle the other.
STAGGERED = {"light_phase": dt.time(7, 0), "dark_phase": dt.time(20, 30)}


at = partial(strategies.at, tz=TZ_NAME)


def timeline(
	start: dt.datetime,
	end: dt.datetime,
	phases: dict,
	start_from: str,
	end_with: str = "light_phase",
) -> Timeline:
	return Timeline(
		start_datetime=start,
		end_datetime=end,
		recording_timezone=TZ,
		phases=phases,
		start_from=start_from,
		end_with=end_with,
	)


# --- resolving the origin ----------------------------------------------------
def test_origin_is_the_first_onset_after_acquisition_begins():
	"""Acquisition starts mid-morning; the experiment starts at that evening's dark onset."""
	line = timeline(at(2023, 5, 24, 9, 30), at(2023, 5, 28, 0, 0), ALIGNED, "dark_phase")
	assert line.experiment_start == at(2023, 5, 24, 20, 0)
	assert line.discarded_lead == dt.timedelta(hours=10, minutes=30)
	assert line.unrecorded_lead == dt.timedelta(0)


def test_origin_equals_acquisition_when_it_begins_on_the_onset():
	"""Starting exactly on the onset trims nothing."""
	line = timeline(at(2023, 5, 24, 20, 0), at(2023, 5, 28, 0, 0), ALIGNED, "dark_phase")
	assert line.experiment_start == at(2023, 5, 24, 20, 0)
	assert line.discarded_lead == dt.timedelta(0)


def test_origin_looks_back_when_the_onset_has_just_passed():
	"""One minute late keeps the day, at the cost of a minute off the first phase.

	Skipping forward to the next onset would forfeit almost 24 hours - which on the
	vGLUT catalog happened to two recordings out of four. The nearest onset is the one
	just behind, so nothing is discarded and the first phase is a minute short instead.
	"""
	line = timeline(at(2023, 5, 24, 20, 1), at(2023, 5, 28, 0, 0), ALIGNED, "dark_phase")

	assert line.experiment_start == at(2023, 5, 24, 20, 0)
	assert line.discarded_lead == dt.timedelta(0)
	assert line.unrecorded_lead == dt.timedelta(minutes=1)


def test_origin_can_be_the_previous_day():
	"""Just after midnight, the nearest onset is the one from the evening before."""
	line = timeline(at(2023, 5, 25, 0, 30), at(2023, 5, 28, 0, 0), ALIGNED, "dark_phase")

	assert line.experiment_start == at(2023, 5, 24, 20, 0)
	assert line.unrecorded_lead == dt.timedelta(hours=4, minutes=30)


def test_leads_are_exclusive_and_measured_as_real_time():
	"""Exactly one lead is ever non-zero, and both are elapsed time, not wall clock.

	This span crosses the spring-forward, where subtracting two datetimes that share a
	tzinfo object would report an hour more than actually elapsed.
	"""
	line = timeline(at(2023, 3, 26, 20, 30), at(2023, 3, 30, 0, 0), ALIGNED, "dark_phase")

	assert line.experiment_start == at(2023, 3, 26, 20, 0)
	assert line.discarded_lead == dt.timedelta(0)
	assert line.unrecorded_lead == dt.timedelta(minutes=30)

	# Day 1 opens at the onset, so a full day later is day 2 in real elapsed time.
	crossing = timeline(at(2023, 3, 25, 20, 0), at(2023, 3, 30, 0, 0), ALIGNED, "dark_phase")
	assert crossing.discarded_lead == dt.timedelta(0)
	assert crossing.unrecorded_lead == dt.timedelta(0)


def test_end_is_the_end_of_the_nearest_end_with_phase():
	"""Acquisition stops mid-dark; the experiment ends at the light onset that closes it."""
	line = timeline(
		at(2023, 5, 24, 20, 0), at(2023, 5, 27, 5, 0), ALIGNED, "dark_phase", "dark_phase"
	)
	assert line.experiment_end == at(2023, 5, 27, 7, 0)
	assert line.unrecorded_tail == dt.timedelta(hours=2)
	assert line.discarded_tail == dt.timedelta(0)


def test_end_trims_data_recorded_past_the_onset():
	line = timeline(
		at(2023, 5, 24, 20, 0), at(2023, 5, 27, 9, 30), ALIGNED, "dark_phase", "dark_phase"
	)
	assert line.experiment_end == at(2023, 5, 27, 7, 0)
	assert line.discarded_tail == dt.timedelta(hours=2, minutes=30)
	assert line.unrecorded_tail == dt.timedelta(0)


def test_end_on_the_onset_trims_nothing():
	line = timeline(
		at(2023, 5, 24, 20, 0), at(2023, 5, 27, 7, 0), ALIGNED, "dark_phase", "dark_phase"
	)
	assert line.experiment_end == at(2023, 5, 27, 7, 0)
	assert line.discarded_tail == line.unrecorded_tail == dt.timedelta(0)


def test_end_with_light_ends_at_the_dark_onset():
	line = timeline(at(2023, 5, 24, 20, 0), at(2023, 5, 27, 18, 0), ALIGNED, "dark_phase")
	assert line.experiment_end == at(2023, 5, 27, 20, 0)


def test_tail_is_measured_in_real_time_across_a_dst_jump():
	"""The wall clock reads an hour more than elapses over the spring-forward."""
	line = partial(
		timeline,
		at(2023, 3, 20, 20, 0),
		phases=ALIGNED,
		start_from="dark_phase",
		end_with="dark_phase",
	)

	assert line(at(2023, 3, 26, 1, 30)).unrecorded_tail == dt.timedelta(hours=4, minutes=30)


def test_analysed_span_runs_from_experiment_start_to_experiment_end():
	"""local_span is the one place the trim happens, so everything downstream inherits it."""
	line = timeline(at(2023, 5, 24, 9, 30), at(2023, 5, 28, 0, 0), ALIGNED, "dark_phase")

	assert line.local_span == (line.experiment_start, line.experiment_end)
	assert line.local_span == (at(2023, 5, 24, 20, 0), at(2023, 5, 27, 20, 0))


def test_days_range_counts_from_the_experiment_start():
	"""The lead-in is not part of day 1, so it does not lengthen the recording."""
	line = timeline(at(2023, 5, 24, 9, 30), at(2023, 5, 28, 20, 0), ALIGNED, "dark_phase")
	# 20:00 on the 24th to 20:00 on the 28th is exactly four 24h days.
	assert line.days_range == (1, 4)


def test_recording_ending_before_the_onset_is_rejected():
	"""A recording that never reaches its start phase holds no experiment."""
	with pytest.raises(ValidationError, match="holds no experiment"):
		timeline(at(2023, 5, 24, 9, 30), at(2023, 5, 24, 18, 0), ALIGNED, "dark_phase")


def test_end_with_must_name_a_defined_phase():
	with pytest.raises(ValidationError, match="end_with"):
		Timeline(
			start_datetime=at(2023, 5, 24, 0, 0),
			end_datetime=at(2023, 5, 28, 0, 0),
			recording_timezone=TZ,
			phases={"light_phase": dt.time(7, 0)},
			start_from="light_phase",
			end_with="light_phase",
		)


def test_end_with_is_required():
	with pytest.raises(ValidationError, match="end_with"):
		Timeline.model_validate(
			{
				"start_datetime": at(2023, 5, 24, 0, 0),
				"end_datetime": at(2023, 5, 28, 0, 0),
				"recording_timezone": TZ,
				"phases": ALIGNED,
				"start_from": "dark_phase",
			}
		)


def test_start_from_must_name_a_defined_phase():
	with pytest.raises(ValidationError, match="start_from"):
		Timeline(
			start_datetime=at(2023, 5, 24, 0, 0),
			end_datetime=at(2023, 5, 28, 0, 0),
			recording_timezone=TZ,
			phases={"light_phase": dt.time(7, 0)},
			start_from="dark_phase",
			end_with="light_phase",
		)


def test_usual_protocol_starts_minutes_before_dark():
	"""The production case: animals go in under lights-on, recording starts just before dark."""
	recording = strategies.analysis_recording(
		tz=TZ_NAME,
		start="2023-05-24 19:55:00",
		finish="2023-05-28 20:00:00",
		phases=ALIGNED,
		start_from="dark_phase",
		end_with="light_phase",
	)
	line = recording.timeline

	assert line.discarded_lead == dt.timedelta(minutes=5)
	# Day 1 hour 0 is dark onset, and the light onset 11 hours later is hour 11.
	frame = pl.DataFrame(
		{
			"datetime": pl.Series([at(2023, 5, 24, 20, 0), at(2023, 5, 25, 7, 0)]).cast(
				pl.Datetime("us", TZ_NAME)
			)
		}
	).with_columns(grids.get_day(recording), grids.get_hour(recording), grids.get_phase(recording))

	assert frame["day"].to_list() == [1, 1]
	assert frame["hour"].to_list() == [0, 11]
	assert frame["phase"].to_list() == ["dark_phase", "light_phase"]


def test_config_loaded_recording_keeps_only_registrations_inside_its_window():
	"""A timeline loaded from JSON holds fixed UTC offsets, not the recording's zone.

	Polars refuses to compare a datetime column with bounds in another zone, so the window
	main_df is trimmed to has to come back from local_span in the recording's own zone.
	"""
	recording = strategies.analysis_recording(
		tz=TZ_NAME,
		start="2023-05-24 19:55:00",
		finish="2023-05-26 20:00:00",
		phases=ALIGNED,
		start_from="dark_phase",
		end_with="light_phase",
	)
	reads = [
		at(2023, 5, 24, 19, 58),  # before the dark onset that starts the experiment
		at(2023, 5, 24, 20, 0),
		at(2023, 5, 25, 12, 0),
		at(2023, 5, 26, 20, 0),
		at(2023, 5, 26, 20, 1),  # after the recording ends
	]
	data = pl.LazyFrame(
		{
			"datetime": reads,
			"antenna": ["1"] * len(reads),
			"time_under": [dt.timedelta(milliseconds=100)] * len(reads),
			"animal_id": ["A"] * len(reads),
			"inserted": [False] * len(reads),
		},
		schema=recording.data_schema,
	)
	loaded = Recording.model_validate({**recording.model_dump(mode="json"), "data": data})

	assert not isinstance(loaded.timeline.end_datetime.tzinfo, ZoneInfo)
	assert build_main_df(loaded, AnalysisParams()).collect()["datetime"].to_list() == reads[1:4]


# --- the grid ----------------------------------------------------------------
def staggered_recording():
	"""Origin on the 20:30 dark onset, so bins sit on the half hour and 07:00 splits one."""
	return strategies.analysis_recording(
		tz=TZ_NAME,
		start="2023-05-24 18:00:00",
		finish="2023-05-27 20:30:00",
		phases=STAGGERED,
		start_from="dark_phase",
		end_with="light_phase",
	)


def aligned_recording():
	return strategies.analysis_recording(
		tz=TZ_NAME,
		start="2023-05-24 18:00:00",
		finish="2023-05-27 20:00:00",
		phases=ALIGNED,
		start_from="dark_phase",
		end_with="light_phase",
	)


def test_aligned_onsets_give_exactly_one_row_per_hour():
	"""With onsets on the hour no bin is ever straddled, so the walk collapses cleanly."""
	recording = aligned_recording()
	grid = grids.build_time_grid(recording).collect()
	span_start, span_end = recording.timeline.local_span

	assert grid.height == int((span_end - span_start) // dt.timedelta(hours=1))
	assert grid.select("day", "hour").is_duplicated().sum() == 0


def test_straddled_bin_gets_one_row_per_phase():
	"""The bin the light onset falls inside is split, so both labels have somewhere to join."""
	grid = grids.build_time_grid(staggered_recording()).collect()
	per_bin = grid.group_by("day", "hour").agg(pl.len().alias("rows"))

	straddled = per_bin.filter(pl.col("rows") > 1)
	assert straddled.height > 0, "expected the 07:00 onset to split a bin anchored at :30"
	assert straddled["rows"].max() == 2

	# The split bin holds one row of each phase, not two of the same.
	split_bin = straddled.row(0, named=True)
	rows = grid.filter((pl.col("day") == split_bin["day"]) & (pl.col("hour") == split_bin["hour"]))
	assert set(rows["phase"].to_list()) == {"light_phase", "dark_phase"}


def test_phase_count_increments_once_across_a_straddled_bin():
	"""Splitting a bin must not double-count the phase change it contains."""
	grid = grids.build_time_grid(staggered_recording()).collect()
	counts = grid["phase_count"].unique().sort().to_list()

	assert counts == list(range(1, len(counts) + 1))
	# Each occurrence is one phase, and consecutive occurrences alternate.
	by_count = grid.group_by("phase_count").agg(pl.col("phase").n_unique().alias("phases"))
	assert by_count["phases"].max() == 1


@pytest.mark.parametrize("phases", [ALIGNED, STAGGERED], ids=["aligned", "staggered"])
def test_every_label_the_data_can_produce_has_a_grid_row(phases):
	"""The invariant the whole grid exists for: no registration can fall outside it.

	A missing row means a null phase_count, which silently drops that registration from
	every reindexed table - the failure mode staggered onsets used to cause.
	"""
	recording = strategies.analysis_recording(
		tz=TZ_NAME,
		start="2023-05-24 18:00:00",
		finish="2023-05-27 03:17:00",  # ends mid-hour as a real recording does
		phases=phases,
		start_from="dark_phase",
		end_with="light_phase",
	)
	span_start, span_end = recording.timeline.local_span

	# Probe every 7 minutes, so instants land inside bins rather than only on edges.
	probes = pl.LazyFrame().select(
		pl.datetime_range(
			span_start, span_end, interval="7m", closed="both", time_zone=TZ_NAME
		).alias("datetime")
	)
	labelled = probes.with_columns(
		grids.get_day(recording), grids.get_phase(recording), grids.get_hour(recording)
	)

	assert grids.assign_phase_count(labelled, recording).collect()["phase_count"].null_count() == 0


def test_origin_inside_a_dst_gap_moves_to_the_clock_that_follows_it():
	"""An onset the spring-forward jump skips still has to name a real instant.

	Warsaw goes 02:00 -> 03:00 on 2023-03-26, so an 02:30 onset that day is a wall
	clock nobody ever saw; left as-is it reports the pre-jump offset and polars
	refuses to build a grid from it.
	"""
	gap = {"light_phase": dt.time(2, 30), "dark_phase": dt.time(20, 0)}
	line = timeline(at(2023, 3, 26, 12, 0), at(2023, 3, 28, 12, 0), gap, "light_phase")

	assert line.experiment_start == dt.datetime(2023, 3, 26, 1, 30, tzinfo=dt.UTC)
	assert line.experiment_start.utcoffset() == dt.timedelta(hours=2)
	assert line.local_span[0] == line.experiment_start

	recording = strategies.analysis_recording(
		tz=TZ_NAME, start="2023-03-26 12:00:00", finish="2023-03-28 12:00:00", phases=gap
	)
	assert grids.build_time_grid(recording).collect().height > 0
