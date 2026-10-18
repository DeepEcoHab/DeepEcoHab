import datetime as dt
from collections.abc import Sequence

import polars as pl

from deepecohab.core import grids, topology, transforms
from deepecohab.core.data_model import (
	CALENDAR_COLUMNS,
	AnalysisParams,
	DataFrameRegistry,
	Layout,
	Recording,
)


@DataFrameRegistry.register("animals")
def build_animals(recording: Recording, params: AnalysisParams) -> pl.LazyFrame:
	"""Frame of the cohort's metadata, one row per animal.

	``animal_id`` carries the same type as in every other table, so this joins onto
	any of them and lets results be grouped by genotype, sex, treatment, mouse line
	or any other recorded attribute.

	Returns:
		One row per animal, with its ``age`` at the start of the recording alongside
		the recorded metadata.
	"""
	recording_start = recording.timeline.local_span[0].date()

	return pl.LazyFrame(
		[
			{
				**animal.model_dump(exclude={"tag"}),
				"animal_id": animal.tag,
				"age": recording_start - animal.date_of_birth,
			}
			for animal in recording.cohort.animals
		],
		schema={
			"animal_id": pl.Enum(recording.cohort.animal_tags),
			"subject_name": pl.String,
			"mouse_line": pl.String,
			"genotype": pl.String,
			"sex": pl.String,
			"date_of_birth": pl.Date,
			"age": pl.Duration("us"),
			"genetic_background": pl.String,
			"treatment": pl.String,
			"notes": pl.String,
		},
	).sort("animal_id")


@DataFrameRegistry.register("main_df")
def build_main_df(recording: Recording, params: AnalysisParams) -> pl.LazyFrame:
	"""Frame of every antenna registration, annotated with where and when it happened.

	This is the table the whole analysis is built on. Positions are resolved first,
	from the antenna each registration crossed and the one before it, and only then is
	the lead-in before the experiment start dropped: trimming first would leave every
	animal's first in-window registration with no predecessor to pair with, so a
	recording with a discarded lead lost that position. The row that survives the trim
	keeps its true position and a ``time_spent`` clipped to the window, so no occupancy
	is attributed to time the analysis does not cover.

	Returns:
		One row per registration, with ``position``, ``time_spent`` and the calendar
		columns added to the raw ``animal_id``, ``antenna``, ``datetime`` and
		``time_under``.
	"""
	start, end = recording.timeline.local_span

	return (
		# One animal cannot be read at two antennas in the same instant; the first read in
		# file order is kept, so (datetime, animal_id) is unique and the row order -
		# padded_df's row_id - is the same on every run.
		recording.data.unique(["animal_id", "datetime"], keep="first", maintain_order=True)
		.sort("datetime", maintain_order=True)
		# Trailing data out first, so the extrapolation anchors on the window and not on it.
		.filter(pl.col("datetime") <= end)
		.pipe(transforms.with_segments, recording.boundaries)
		.pipe(
			transforms.extrapolate_last_position,
			recording.boundaries,
			end,
			params.extrapolation_limit,
		)
		.pipe(transforms.calculate_time_spent)
		.pipe(transforms.get_animal_position, recording.layout.antenna_combinations)
		.with_columns(
			pl.when(pl.col("__undefined"))
			.then(pl.lit(Layout.UNDEFINED))
			.otherwise(pl.col("position"))
			.cast(pl.Categorical)
			.alias("position")
		)
		.filter(pl.col("datetime") >= start)
		.with_columns(
			pl.min_horizontal("time_spent", pl.col("datetime") - pl.lit(start)).alias("time_spent")
		)
		# The window is half-open, so a row on its end (the carried tail, say) still belongs
		# to the last hour rather than opening one the grid does not have.
		.with_columns(
			pl.min_horizontal("datetime", pl.lit(end - dt.timedelta(microseconds=1))).alias(
				"__label"
			)
		)
		.with_columns(
			grids.get_phase(recording, "__label"),
			grids.get_day(recording, "__label"),
			grids.get_hour(recording, "__label"),
		)
		.pipe(grids.assign_phase_count, recording)
		.drop("__undefined", "__segment", "__label")
		.sort("datetime", "animal_id")
	)


@DataFrameRegistry.register("time_grid")
def build_time_grid(recording: Recording, params: AnalysisParams) -> pl.LazyFrame:
	"""Frame of every cell the recording's tables are binned into.

	``chasings_df`` and ``pairwise_meetings`` store only the cells something happened
	in, so a reader averaging over cells zero-fills them against this.
	"""
	return grids.build_time_grid(recording)


@DataFrameRegistry.register("padded_df", requires=["main_df"])
def build_padded_df(recording: Recording, params: AnalysisParams) -> pl.LazyFrame:
	"""Frame of ``main_df`` visits cut wherever a cell of the time grid begins.

	Occupancy has to be attributed to the hour, phase and day it happened in, so a
	visit spanning several cells is split into one piece per cell. Tables that measure
	time rather than events read this instead of ``main_df``.
	"""
	pieces = transforms.split_visits_on_cells(recording.load_results("main_df"), recording)

	# A piece is dated by its own start, not by the end of the visit it came from, so
	# the phase_count inherited from that visit no longer applies. Every piece starts
	# inside the window - build_main_df clips the first visit's time_spent to it - so
	# this left join always resolves and no piece leaves with a null phase_count.
	return grids.assign_phase_count(pieces.drop("phase_count"), recording).sort(
		"datetime", "row_id"
	)


@DataFrameRegistry.register("phase_durations")
def build_phase_durations(recording: Recording, params: AnalysisParams) -> pl.LazyFrame:
	"""Frame with how long each phase occurrence of the recording lasted.

	Durations are accurate to roughly one minute, which is the resolution the
	recording window is walked at.

	Returns:
		One row per phase occurrence, with ``phase``, ``phase_count`` and
		``duration``.
	"""
	start, end = recording.timeline.local_span

	return (
		pl.LazyFrame()
		.select(
			pl.datetime_range(
				start,
				end,
				interval="1m",
				closed="left",
				time_zone=recording.timeline.recording_timezone.key,
			)
			.dt.round("1m")
			.alias("datetime")
		)
		.with_columns(
			grids.get_phase(recording), grids.get_day(recording), grids.get_hour(recording)
		)
		.pipe(grids.assign_phase_count, recording)
		.group_by("phase", "phase_count")
		.agg(
			((pl.col("datetime").max() - pl.col("datetime").min()) + dt.timedelta(minutes=1)).alias(
				"duration"
			)
		)
	)


@DataFrameRegistry.register("event_bouts")
def build_event_bouts(recording: Recording, params: AnalysisParams) -> pl.LazyFrame:
	"""Frame placing every bout of the recording's events on its calendar.

	A bout gets one row for each hourly cell of the time grid it overlaps, so the table
	joins onto any analysis table by the calendar columns, and a plot finds the bins a
	bout covers with a filter. A bout ending on the hour does not reach into the next.

	Returns:
		One row per bout, position and calendar cell, with the bout's ``event``,
		``position`` (null for the whole habitat), ``start`` and ``end``. Empty when the
		recording declares no events.
	"""
	bouts = pl.LazyFrame(
		[
			{"event": event.name, "position": bout.position, "start": bout.start, "end": bout.end}
			for event in recording.events
			for bout in event.bouts
		],
		schema={
			"event": pl.Enum([event.name for event in recording.events]),
			"position": pl.List(pl.String),
			"start": pl.Datetime("us", time_zone=recording.timeline.recording_timezone.key),
			"end": pl.Datetime("us", time_zone=recording.timeline.recording_timezone.key),
		},
	)

	return (
		bouts.explode("position", empty_as_null=True)
		.with_columns(
			pl.datetime_ranges(
				pl.col("start").dt.truncate("1m"),
				(pl.col("end") - pl.duration(microseconds=1)).dt.truncate("1m"),
				interval="1m",
			).alias("minute")
		)
		.explode("minute", empty_as_null=True)
		.with_columns(
			grids.get_phase(recording, "minute"),
			grids.get_day(recording, "minute"),
			grids.get_hour(recording, "minute"),
		)
		.drop("minute")
		.unique()
		.pipe(grids.assign_phase_count, recording)
		.select("event", "position", "start", "end", *CALENDAR_COLUMNS)
		.sort("start", "event", "position", "day", "hour", "phase_count", nulls_last=False)
	)


def event_status(recording: Recording, event_names: Sequence[str]) -> pl.LazyFrame:
	"""Per ``(day, hour)`` status of ``event_names``, for one recording's calendar.

	Not a registered step: ``Project.generate_project_table`` calls this directly, once
	it knows the union of event names declared anywhere in the project, so every
	recording's frame carries the same event columns before the vertical concat.

	A bout marks every calendar hour it touches, matching ``event_bouts``. A cell a
	bout itself covers reads ``"During"``; one sharing the bout's hour-of-day on a
	different day reads ``"Same hours, other days"``; anything else this recording
	declares the event on reads ``"Other hours"``. A name this recording does not
	declare - including ``"Any event"`` when it declares nothing at all - is null
	throughout: the recording is left out of the event, not counted as outside it.

	Args:
		recording: the recording whose calendar and ``event_bouts`` this reads.
		event_names: columns to produce; ``"Any event"`` matches every declared event.

	Returns:
		``day``, ``hour`` and one ``String`` column per name in ``event_names``.
	"""
	grid = grids.build_time_grid(recording).select("day", "hour").unique()
	declared = {event.name for event in recording.events}
	bouts = recording.load_results("event_bouts") if declared else None

	def status_column(name: str, name_bouts: pl.LazyFrame) -> pl.LazyFrame:
		during = name_bouts.select("day", "hour").unique().with_columns(_during=pl.lit(True))
		same_hour = name_bouts.select("hour").unique().with_columns(_same_hour=pl.lit(True))

		return (
			grid.join(during, on=["day", "hour"], how="left")
			.join(same_hour, on="hour", how="left")
			.select(
				"day",
				"hour",
				pl.when(pl.col("_during").is_not_null())
				.then(pl.lit("During"))
				.when(pl.col("_same_hour").is_not_null())
				.then(pl.lit("Same hours, other days"))
				.otherwise(pl.lit("Other hours"))
				.alias(name),
			)
		)

	result = grid
	for name in event_names:
		if bouts is None or (name != "Any event" and name not in declared):
			result = result.with_columns(pl.lit(None, dtype=pl.String).alias(name))
			continue

		name_bouts = bouts if name == "Any event" else bouts.filter(pl.col("event") == name)
		result = result.join(status_column(name, name_bouts), on=["day", "hour"], how="left")

	return result


def _routes_frame(routes: dict[tuple[str, str], list[str]]) -> pl.LazyFrame:
	"""``topology.unique_routes`` as a frame joinable on the step it describes."""
	return pl.LazyFrame(
		[(source, target, route) for (source, target), route in routes.items()],
		schema={
			"previous_antenna": pl.Categorical(),
			"antenna": pl.Categorical(),
			"crossed": pl.List(pl.Categorical()),
		},
		orient="row",
	)


@DataFrameRegistry.register("recording_quality")
def build_recording_quality(recording: Recording, params: AnalysisParams) -> pl.LazyFrame:
	"""Frame of how much of each animal's movement the antennas actually caught.

	Every pass is ``correct`` (read), ``interpolated`` (inserted by acquisition
	preprocessing) or ``bad``. An animal that turns up at an antenna the layout does
	not join to the one before it passed antennas that never fired and preprocessing
	did not fill in; every such step is charged as bad to the antennas the animal must
	have crossed, which is what makes the rate one per antenna rather than one per
	recording: a column that stands out is a weak antenna, a row a weak transponder.
	A step across a recording boundary is not a pass - nothing was recording.

	Returns:
		One row per animal and antenna, with the ``correct``, ``interpolated`` and
		``bad`` pass counts inside the analysed window and ``miss_rate``, the
		percentage of that animal's passes over that antenna the antenna did not read.
		The rate is a lower bound.
	"""
	start, end = recording.timeline.local_span
	antenna_combinations = recording.layout.antenna_combinations

	steps = (
		recording.data.filter(pl.col("datetime").is_between(start, end))
		.sort("datetime")
		.pipe(transforms.with_segments, recording.boundaries)
		.with_columns(
			pl.col("antenna").shift(1).over("animal_id", "__segment").alias("previous_antenna")
		)
		.join(
			_routes_frame(topology.unique_routes(antenna_combinations)),
			on=["previous_antenna", "antenna"],
			how="left",
		)
	)

	read = steps.group_by("animal_id", "antenna").agg(
		(~pl.col("inserted")).sum().alias("correct"), pl.col("inserted").sum().alias("interpolated")
	)
	bad = (
		steps.filter(pl.col("crossed").is_not_null())
		.explode("crossed")
		.group_by("animal_id", pl.col("crossed").alias("antenna"))
		.len("bad")
	)
	passes = pl.sum_horizontal("correct", "interpolated", "bad")

	antennas = sorted(topology.antennas(antenna_combinations))
	grid = grids.build_animal_grid(recording, "animal_id").join(
		pl.LazyFrame({"antenna": antennas}, schema={"antenna": pl.Categorical()}), how="cross"
	)

	return (
		grid.join(read, on=["animal_id", "antenna"], how="left")
		.join(bad, on=["animal_id", "antenna"], how="left")
		.with_columns(pl.col("correct", "interpolated", "bad").fill_null(0).cast(pl.UInt32))
		.with_columns(
			pl.when(passes > 0)
			.then(100 * (pl.col("interpolated") + pl.col("bad")) / passes)
			.otherwise(0.0)
			.alias("miss_rate")
		)
		.sort("animal_id", "antenna")
	)
