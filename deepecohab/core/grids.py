import datetime as dt
from itertools import product
from typing import Any

import polars as pl
import polars.selectors as cs

from deepecohab.core.data_model import CALENDAR_COLUMNS, Recording


def get_phase(recording: Recording, datetime_column: str = "datetime") -> pl.Expr:
	"""Expression assigning each row a circadian phase from its local time of day.

	``timeline.phases`` maps each phase name to the wall-clock time it begins, e.g.
	``{"light_phase": 07:00, "dark_phase": 20:00}`` means light_phase runs
	[07:00, 20:00) and dark_phase runs [20:00, 07:00), wrapping around midnight.
	The phases are assumed to tile the full 24h day.
	"""
	phases = recording.timeline.phases
	boundaries = sorted((start, name) for name, start in phases.items())

	# Anchored on the experiment start, as get_day and get_hour are, so a phase label
	# depends only on the instant and not on which frame it arrives in.
	origin_daylight = recording.timeline.experiment_start.dst() or dt.timedelta(0)
	daylight_shift = pl.col(datetime_column).dt.dst_offset() - pl.lit(origin_daylight)
	time_of_day = (pl.col(datetime_column) - daylight_shift).dt.time()

	expression = pl.lit(boundaries[-1][1])
	for start, name in boundaries:
		expression = pl.when(time_of_day >= start).then(pl.lit(name)).otherwise(expression)

	return expression.cast(pl.Enum(list(phases))).alias("phase")


def _elapsed_hours(recording: Recording, datetime_column: str) -> pl.Expr:
	"""Expression counting whole hours from the experiment start to each row."""
	origin = pl.lit(recording.timeline.experiment_start)
	return (pl.col(datetime_column) - origin).dt.total_hours()


def get_day(recording: Recording, datetime_column: str = "datetime") -> pl.Expr:
	"""Expression numbering each row's experiment day, counting from 1 at the start.

	A day is 24 hours measured from the experiment start, not a calendar date.
	"""
	return (_elapsed_hours(recording, datetime_column) // 24 + 1).cast(pl.UInt16).alias("day")


def get_hour(recording: Recording, datetime_column: str = "datetime") -> pl.Expr:
	"""Expression giving each row's hour within its experiment day, 0 to 23."""
	return (_elapsed_hours(recording, datetime_column) % 24).cast(pl.UInt8).alias("hour")


def assign_phase_count(frame: pl.LazyFrame, recording: Recording) -> pl.LazyFrame:
	"""Frame with ``phase_count`` attached, looked up from the time grid.

	Successive phase occurrences are numbered chronologically regardless of phase
	type, so the first phase of the recording is 1, the next (of either type) is 2,
	and so on; a lower count always came earlier.

	Only ``main_df`` needs this - every table built from it carries the column
	through. ``frame`` must already carry ``day``, ``phase`` and ``hour``.
	"""
	grid_counts = build_time_grid(recording).select(CALENDAR_COLUMNS).unique()

	return frame.join(grid_counts, on=["day", "phase", "hour"], how="left")


def build_time_grid(recording: Recording) -> pl.LazyFrame:
	"""Frame with one row per hourly bin spanning the recording.

	This is the temporal half of the dense reference grid. Cross-join it with
	:func:`build_animal_grid` to obtain the full grid for a given table.

	A bin that a phase boundary falls inside holds two phases, and gets one row for
	each, so every label a registration can carry has a row to join onto.

	Returns:
		Frame with columns ``day``, ``phase``, ``hour`` and ``phase_count``.
	"""
	return _time_cells(recording).drop("bin_start")


def cell_starts(recording: Recording) -> pl.Series:
	"""Every instant a cell of :func:`build_time_grid` begins, in order.

	Cutting an interval at these instants leaves no piece crossing from one hour, phase
	or day into the next, so each piece belongs wholly to the cell it starts in.
	"""
	return _time_cells(recording).select("bin_start").collect().to_series()


def _time_cells(recording: Recording) -> pl.LazyFrame:
	"""The time grid's cells in order, each with the ``bin_start`` instant it opens at."""
	timezone = recording.timeline.recording_timezone
	start, finish = recording.timeline.local_span

	return (
		pl.LazyFrame()
		.select(
			pl.datetime_range(
				start, finish, interval="1m", closed="left", time_zone=timezone.key
			).alias("minute")
		)
		.with_columns(
			get_day(recording, "minute"),
			get_phase(recording, "minute"),
			get_hour(recording, "minute"),
		)
		# Walking by minute rather than by hour is what separates the two phases of a
		# straddled bin; collapsing to the distinct labels gives back the hourly grid.
		.group_by("day", "phase", "hour")
		.agg(pl.min("minute").alias("bin_start"))
		.sort("bin_start")
		# The grid holds every phase occurrence in chronological order, so numbering its
		# runs directly is correct here - and only here.
		.with_columns((pl.col("phase").rle_id() + 1).cast(pl.UInt16).alias("phase_count"))
	)


def build_animal_grid(
	recording: Recording,
	columns: str | tuple[str, str],
	ordered: bool = True,
	positions: list[str] | None = None,
) -> pl.LazyFrame:
	"""Frame enumerating every animal, or every animal pair, of the cohort.

	This is the entity half of the reference grid. Cross-join it with
	:func:`build_time_grid` to obtain the dense grid a table is reindexed onto.

	Args:
		recording: the recording whose cohort the grid covers.
		columns: a single column name for a per-animal grid (e.g. ``"animal_id"``),
			or a pair of names for a per-pair grid (e.g. ``("winner", "loser")`` or
			``("animal_id", "animal_id_2")``).
		ordered: for pair grids only, whether to enumerate ordered pairs (``a != b``
			in both directions, for directed measures like chasings) or unordered
			combinations (``a < b``, for symmetric measures like pairwise
			encounters). Ignored for single-animal grids.
		positions: when given, the grid is crossed with these positions under a
			``position`` column. When ``None`` no position column is added.

	Returns:
		Frame with the requested animal column(s) and, if ``positions`` is given, a
		``position`` column.
	"""
	cohort = recording.cohort
	animals = pl.Enum(cohort.animal_tags)
	schema: dict[str, Any] = {}

	if isinstance(columns, str):
		schema[columns] = animals
		rows: list[tuple] = [(animal,) for animal in cohort.animal_tags]
	else:
		first, second = columns
		rows = cohort.animal_product if ordered else cohort.animal_combinations
		schema[first] = schema[second] = animals

	if positions is not None:
		rows = [(*row, position) for row, position in product(rows, positions)]
		schema["position"] = pl.Categorical

	return pl.LazyFrame(rows, schema=schema, orient="row")


def reindex_onto_grid(
	data: pl.LazyFrame,
	recording: Recording,
	columns: str | tuple[str, str],
	ordered: bool = True,
	positions: list[str] | None = None,
) -> pl.LazyFrame:
	"""Frame ``data`` reindexed onto the dense time x animal (x position) grid.

	Cross-joins :func:`build_time_grid` with :func:`build_animal_grid`, then
	left-joins ``data`` onto the result so every grid cell is present, filling cells
	with no observed value with ``0``. The join keys are the time columns
	(``phase``, ``day``, ``phase_count``, ``hour``), the ``position`` column when
	``positions`` is given, and the animal column(s) named in ``columns``.

	Args:
		data: observed table to reindex; must contain the join columns.
		recording: the recording the grid is built for.
		columns: animal column(s) for :func:`build_animal_grid`.
		ordered: for pair grids, whether to enumerate ordered pairs. See
			:func:`build_animal_grid`.
		positions: when given, the grid is crossed with these positions and
			``position`` becomes a join key.
	"""
	full_grid = build_time_grid(recording).join(
		build_animal_grid(recording, columns, ordered=ordered, positions=positions),
		how="cross",
	)

	animal_columns = [columns] if isinstance(columns, str) else list(columns)
	join_columns = [*CALENDAR_COLUMNS]
	if positions is not None:
		join_columns.append("position")
	join_columns += animal_columns

	return zero_filled(full_grid, data, join_columns)


def zero_filled(scaffold: pl.LazyFrame, data: pl.LazyFrame, on: list[str]) -> pl.LazyFrame:
	"""``data`` left-joined onto every row of ``scaffold``, with zero where it has none."""
	return (
		scaffold.join(data, on=on, how="left")
		.with_columns(cs.duration().fill_null(pl.duration(microseconds=0)))
		.fill_null(0)
	)
