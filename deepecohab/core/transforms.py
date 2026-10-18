import datetime as dt
from collections.abc import Sequence

import polars as pl
import polars.selectors as cs

from deepecohab.core import grids
from deepecohab.core.data_model import Boundary, Layout, Recording


def csv_ready(frame: pl.DataFrame) -> pl.DataFrame:
	"""``frame`` with every Duration column in seconds and every Enum/Categorical as a string.

	CSV has neither type, so a plain ``write_csv`` would otherwise dump raw microsecond
	integers for a duration and silently drop a category's declared order.
	"""
	return frame.with_columns(
		cs.duration().dt.total_microseconds() / 1_000_000,
		(cs.enum() | cs.categorical()).cast(pl.String),
	)


def calculate_time_spent(frame: pl.LazyFrame) -> pl.LazyFrame:
	"""Frame with ``time_spent``: the gap to the same animal's previous registration.

	This is how long the animal spent in the state the row describes, held as a
	duration so that arithmetic against ``datetime`` stays exact. The first
	registration of each animal has no predecessor and gets zero.
	"""
	time_spent = (
		(pl.col("datetime") - pl.col("datetime").shift(1))
		.over("animal_id")
		.cast(pl.Duration("us"))
		.fill_null(pl.duration(microseconds=0))
	)
	return frame.with_columns(time_spent.alias("time_spent"))


def get_animal_position(frame: pl.LazyFrame, antenna_combinations: dict[str, str]) -> pl.LazyFrame:
	"""Frame with ``position`` derived from each row's antenna and the previous one.

	Per animal, the previous and current antenna form a ``"<previous>_<current>"``
	key, which ``antenna_combinations`` resolves to a cage or a directional tunnel. Pairs the
	map does not name become ``"undefined"``, as does the first reading of each
	animal, which has no predecessor to pair with.
	"""
	# Filled after the cast, and with a non-numeric sentinel, so the unpaired first
	# reading cannot collide with a layout that numbers an antenna 0.
	previous_antenna = (
		pl.col("antenna").shift(1).over("animal_id").cast(pl.Utf8).fill_null("missing")
	)
	current_antenna = pl.col("antenna").cast(pl.Utf8)
	antenna_pair = pl.concat_str([previous_antenna, pl.lit("_"), current_antenna])

	return frame.with_columns(
		antenna_pair.replace_strict(antenna_combinations, default=Layout.UNDEFINED)
		.alias("position")
		.cast(pl.Categorical)
	)


def with_segments(frame: pl.LazyFrame, boundaries: Sequence[Boundary]) -> pl.LazyFrame:
	"""Frame with ``__segment``: how many recording boundaries lie before each row.

	The rows of one segment were recorded without a stop between them, so an animal's
	track is only continuous within one.
	"""
	after = [(pl.col("datetime") >= boundary.end).cast(pl.UInt32) for boundary in boundaries]
	return frame.with_columns(pl.sum_horizontal(pl.lit(0, pl.UInt32), *after).alias("__segment"))


def extrapolate_last_position(
	frame: pl.LazyFrame, boundaries: Sequence[Boundary], end: dt.datetime, limit: float
) -> pl.LazyFrame:
	"""Frame with each animal's last position carried on towards the end of its segment.

	This gives a better estimate of time spent in positions and of cage occupancy in
	recordings with low activity, but silence is only weak evidence of staying put: the
	carry stops ``limit`` seconds after the last registration, and an animal silent
	past that gets a second row at the segment's end marked ``__undefined`` for the
	caller to turn into :attr:`Layout.UNDEFINED`. Targeting the window end rather than
	the last read in the data is what keeps a recording's final stretch attributed at all.

	A recording boundary ends a segment at its start, the last record before the stop.
	Where the animal was while nothing recorded is unknown, so its first row after the
	boundary is marked ``__undefined`` too: that row's interval runs from the stop to
	the read, and the antenna pair it would resolve spans the stop.

	Both added rows repeat the last antenna, so :func:`get_animal_position` resolves the
	cap row as the self-pair - the cage at that antenna, where the animal *is* - rather
	than the position the last row named, which is where it was coming from.

	Args:
		frame: registrations, with ``animal_id``, ``antenna``, ``datetime`` and the
			``__segment`` of :func:`with_segments` over ``boundaries``.
		boundaries: the recording's boundaries, in order.
		end: the analysed window's end, which the last segment's carry and tail target.
		limit: how many seconds the last position may be carried for.

	Returns:
		``frame`` plus the added rows, with an ``__undefined`` flag the caller consumes.
	"""
	schema = frame.collect_schema()
	columns = [*schema.names(), "__undefined"]
	segment_ends = pl.LazyFrame(
		{
			"__segment": range(len(boundaries) + 1),
			"__end": [min(boundary.start, end) for boundary in boundaries] + [end],
		},
		schema={"__segment": pl.UInt32, "__end": schema["datetime"]},
	)

	last_rows = (
		frame.group_by("animal_id", "__segment")
		.agg(pl.all().sort_by("datetime").last())
		.join(segment_ends, on="__segment")
	)
	capped = last_rows.with_columns(
		pl.min_horizontal(pl.col("datetime") + pl.duration(seconds=limit), "__end").alias("__cap")
	)

	carried = capped.filter(pl.col("__cap") > pl.col("datetime")).with_columns(
		pl.col("__cap").alias("datetime"), pl.lit(False).alias("__undefined")
	)
	tail = capped.filter(pl.col("__end") > pl.col("__cap")).with_columns(
		pl.col("__end").alias("datetime"), pl.lit(True).alias("__undefined")
	)
	# Every read at the first instant, not the first row: the sort below does not keep
	# ties in order, and a tie left unmarked could inherit the stop's interval.
	restart = (pl.col("__segment") > 0) & (
		pl.col("datetime") == pl.col("datetime").min().over("animal_id", "__segment")
	)

	return pl.concat(
		[
			frame.with_columns(restart.alias("__undefined")),
			carried.select(columns),
			tail.select(columns),
		]
	).sort("datetime")


def remove_tunnel_directionality(frame: pl.LazyFrame, recording: Recording) -> pl.LazyFrame:
	"""Frame with directional tunnel positions mapped to their undirected names."""
	return frame.with_columns(
		pl.col("position").cast(pl.Utf8).replace(recording.layout.tunnels_map).cast(pl.Categorical)
	)


def add_occupancy_bounds(frame: pl.LazyFrame) -> pl.LazyFrame:
	"""Frame with ``start`` and ``end`` bounding each occupancy interval.

	``datetime`` is the interval end, so ``start`` is ``datetime - time_spent``.
	"""
	return frame.with_columns(
		pl.col("datetime").alias("end"),
		(pl.col("datetime") - pl.col("time_spent")).alias("start"),
	)


def with_animal_bits(frame: pl.LazyFrame, animal: str) -> tuple[pl.LazyFrame, pl.LazyFrame]:
	"""Frame with a single-bit ``bit`` mask per animal, and the lookup that decodes it.

	Summed over the animals present, the bits form a presence bitmask (one bit per
	animal, so capped at ~63 animals - far above any EcoHab study).

	Args:
		frame: table with an animal id column.
		animal: name of that column.

	Returns:
		``frame`` with ``bit`` added, and the ``animal``/``bit`` pairs that map each
		bit back to its id.
	"""
	bits = frame.with_columns(
		pl.lit(2, dtype=pl.Int64)
		.pow(pl.col(animal).to_physical().cast(pl.Int64))
		.cast(pl.Int64)
		.alias("bit")
	)
	return bits, bits.select(animal, "bit").unique()


def occupancy_spans(intervals: pl.LazyFrame) -> pl.LazyFrame:
	"""Frame of every span of time a position was occupied, one row per animal inside.

	A sweep over the occupancy intervals turns each stay into an enter and a leave
	event, so at every instant the running count says how many animals the position
	holds and the running signed bit-sum says which ones (one bit per animal, so the
	cumulative sum is a true presence bitmask, capped at ~63 animals - far above any
	EcoHab study). Every arrival and departure closes a span and opens the next, so a
	span holds one fixed set of animals, recovered by decoding its bitmask.

	Args:
		intervals: table with ``animal_id``, ``position`` and the ``start``/``end``
			bounds :func:`add_occupancy_bounds` adds.

	Returns:
		One row per span and animal inside it, with ``position``, ``time``, ``end`` and
		``animals_present``.
	"""
	codes, animal_lookup = with_animal_bits(
		intervals.select("animal_id", "position", "start", "end"), "animal_id"
	)

	enters = codes.select(
		pl.col("start").alias("time"),
		"position",
		pl.lit(1, dtype=pl.Int32).alias("delta"),
		pl.col("bit").alias("bit_delta"),
	)
	leaves = codes.select(
		pl.col("end").alias("time"),
		"position",
		pl.lit(-1, dtype=pl.Int32).alias("delta"),
		pl.col("bit").neg().alias("bit_delta"),
	)

	events = (
		pl.concat([enters, leaves])
		.sort("position", "time")
		.with_columns(
			pl.col("delta").cum_sum().over("position").alias("animals_present"),
			pl.col("bit_delta").cum_sum().over("position").alias("mask"),
			pl.col("time").shift(-1).over("position").alias("next_time"),
		)
	)

	spans = events.filter(
		pl.col("next_time").is_not_null(),
		pl.col("animals_present") >= 1,
		pl.col("next_time") > pl.col("time"),
	).select("position", "time", pl.col("next_time").alias("end"), "animals_present", "mask")

	return (
		spans.join(animal_lookup, how="cross")
		.filter((pl.col("mask") & pl.col("bit")) != 0)
		.select("position", "time", "end", "animals_present", "animal_id")
	)


def contiguous_run_id(start: str, end: str, *partition: str) -> pl.Expr:
	"""Expression numbering runs of back-to-back intervals within each partition.

	A row starting after the previous row's end opens a new run; one starting at or
	before it continues the run, so an overlap counts as back-to-back. Rows must be
	sorted by ``start`` within each partition.
	"""
	return (pl.col(start) > pl.col(end).shift(1)).fill_null(True).cum_sum().over(partition)


def merge_contiguous_stays(intervals: pl.LazyFrame) -> pl.LazyFrame:
	"""Frame of each animal's back-to-back intervals in one position merged into one stay.

	A zero-length interval is dropped first: nothing can happen in it, and left in it
	would split the stay it sits inside.

	Args:
		intervals: table with ``animal_id``, ``position`` and the ``start``/``end``
			bounds :func:`add_occupancy_bounds` adds.

	Returns:
		One row per stay, with ``animal_id``, ``position``, ``start`` and ``end``.
	"""
	# Overlapping counts as back-to-back: registrations never overlap, but intervals
	# rounded to the microsecond can, and two stays of one animal would meet each other's
	# partners twice over the overlap.
	return (
		intervals.filter(pl.col("end") > pl.col("start"))
		.sort("animal_id", "position", "start")
		.with_columns(contiguous_run_id("start", "end", "animal_id", "position").alias("__stay"))
		.group_by("animal_id", "position", "__stay")
		.agg(pl.col("start").min(), pl.col("end").max())
		.drop("__stay")
	)


def overlapping_stays(stays: pl.LazyFrame) -> pl.LazyFrame:
	"""Frame of every meeting: two animals' stays overlapping in one position.

	A meeting runs from the later arrival to the earlier departure, so it is one unbroken
	stretch of co-presence however many other animals come and go during it, and the
	result holds one row per meeting rather than one per pair and moment.

	A sweep rather than an overlap join: with the stays sorted by position and start,
	every stay that starts before another ends, and after it starts, overlaps it, so one
	``search_sorted`` over a composite ``(position, time)`` key bounds each stay's
	partners and no pair that fails to overlap is ever formed. Row order is unspecified.

	Args:
		stays: :func:`merge_contiguous_stays` output.

	Returns:
		One row per meeting, with ``position``, ``animal_id`` before ``animal_id_2``,
		``start`` and ``end``.
	"""
	collected = stays.collect()
	if collected.is_empty():
		return collected.lazy().select(
			"position",
			"animal_id",
			pl.col("animal_id").alias("animal_id_2"),
			"start",
			"end",
		)

	origin = collected["start"].min()
	# One position per block of the key, wide enough that no time spills into the next.
	block = collected.select(
		(pl.col("end").max() - pl.col("start").min()).dt.total_microseconds() + 1
	).item()

	def key(column: str) -> pl.Expr:
		return pl.col("position").to_physical().cast(pl.Int64) * block + (
			(pl.col(column) - origin).dt.total_microseconds()
		)

	ordered = collected.sort(pl.col("position").to_physical(), "start").with_row_index("index")
	ordered = ordered.with_columns(key("start").alias("__start_key"), key("end").alias("__end_key"))
	# Stays at or after this one in start order that begin before it ends.
	stop = ordered["__start_key"].search_sorted(ordered["__end_key"], side="left")
	partners = ordered.select(
		pl.col("index").alias("partner"),
		pl.col("animal_id").alias("animal_id_2"),
		pl.col("start").alias("partner_start"),
		pl.col("end").alias("partner_end"),
	)

	return (
		ordered.lazy()
		.with_columns(pl.int_ranges(pl.col("index") + 1, stop).alias("partner"))
		.explode("partner")
		.drop_nulls("partner")
		.join(partners.lazy(), on="partner")
		.filter(pl.col("animal_id") != pl.col("animal_id_2"))
		.select(
			"position",
			pl.when(pl.col("animal_id") < pl.col("animal_id_2"))
			.then("animal_id")
			.otherwise("animal_id_2")
			.alias("animal_id"),
			pl.when(pl.col("animal_id") < pl.col("animal_id_2"))
			.then("animal_id_2")
			.otherwise("animal_id")
			.alias("animal_id_2"),
			# The partner starts no earlier, so it is the later arrival.
			pl.col("partner_start").alias("start"),
			pl.min_horizontal("end", "partner_end").alias("end"),
		)
	)


def cooccupancy_spans(intervals: pl.LazyFrame) -> pl.LazyFrame:
	"""Frame of every span of time two animals shared a position, one row per pair.

	The occupied spans of :func:`occupancy_spans` holding two or more animals, paired
	up. A pair's co-presence is cut into a new span whenever any third animal arrives
	or leaves, so contiguous spans of one pair are numbered with a single ``meeting_id``
	and read as one meeting.

	Args:
		intervals: table with ``animal_id``, ``position`` and the ``start``/``end``
			bounds :func:`add_occupancy_bounds` adds.

	Returns:
		One row per span and pair, with ``position``, ``time``, ``end``, ``animal_id``,
		``animal_id_2`` and ``meeting_id`` numbering meetings within the pair and position.
	"""
	present = occupancy_spans(intervals).filter(pl.col("animals_present") >= 2)

	pairs = present.join(present, on=["position", "time", "end"], suffix="_2").filter(
		pl.col("animal_id") < pl.col("animal_id_2")
	)

	group = ["animal_id", "animal_id_2", "position"]
	return (
		pairs.select("position", "time", "end", "animal_id", "animal_id_2")
		.sort(*group, "time")
		.with_columns(contiguous_run_id("time", "end", *group).alias("meeting_id"))
	)


def split_on_cell_boundaries(
	frame: pl.LazyFrame, recording: Recording, start: str, end: str
) -> pl.LazyFrame:
	"""Frame with every ``[start, end)`` interval cut wherever a cell of the time grid begins.

	No piece then crosses from one hour, phase or day into the next, so its time can be
	attributed to the cell it lies in. A zero-length interval stays a single piece.

	Args:
		frame: table holding the interval bounds.
		recording: the recording whose grid the cuts come from.
		start: the column holding each interval's start.
		end: the column holding each interval's end.

	Returns:
		One row per piece, ``start`` and ``end`` narrowed to it, with ``first_piece``
		marking the piece each interval opens with and the ``phase``, ``day`` and
		``hour`` of the cell it lies in.
	"""
	starts = grids.cell_starts(recording)
	cells = pl.lit(starts)
	last = len(starts) - 1

	# Indices into the cell starts: the cell holding the interval's start (-1 before the
	# window) and the last cell the interval reaches into, never before the first.
	first = cells.search_sorted(pl.col(start), side="right").cast(pl.Int64) - 1
	final = pl.max_horizontal(
		cells.search_sorted(pl.col(end), side="left").cast(pl.Int64) - 1, first
	)

	return (
		frame.with_columns(first.alias("__first"), final.alias("__final"))
		.with_columns(pl.int_ranges("__first", pl.col("__final") + 1).alias("__cell"))
		.explode("__cell")
		.with_columns(
			pl.when(pl.col("__cell") == pl.col("__first"))
			.then(pl.col(start))
			.otherwise(cells.gather(pl.col("__cell").clip(0, last)))
			.alias(start),
			pl.when(pl.col("__cell") == pl.col("__final"))
			.then(pl.col(end))
			.otherwise(cells.gather((pl.col("__cell") + 1).clip(0, last)))
			.alias(end),
			(pl.col("__cell") == pl.col("__first")).alias("first_piece"),
		)
		.with_columns(
			grids.get_phase(recording, start),
			grids.get_day(recording, start),
			grids.get_hour(recording, start),
		)
		.drop("__first", "__final", "__cell")
	)


def split_visits_on_cells(frame: pl.LazyFrame, recording: Recording) -> pl.LazyFrame:
	"""Frame with visits that straddle cells of the time grid cut into one piece per cell.

	Each row is the interval ``[datetime - time_spent, datetime)``, cut by
	:func:`split_on_cell_boundaries` so its time is attributed to the right hour, phase
	and day. ``time_spent`` becomes each piece's own length and ``time_under`` is shared
	out in proportion.

	Within a split visit only the first piece keeps ``interpolated=False``, so
	filtering ``interpolated == False`` recovers one row per original visit.

	Args:
		frame: table with ``datetime``, ``time_spent`` and ``time_under``.
		recording: the recording whose grid the pieces are assigned to.

	Returns:
		One row per piece, with ``interpolated`` and a ``row_id`` identifying the visit
		the piece came from.
	"""
	visit = pl.col("time_spent").dt.total_microseconds()
	piece = (pl.col("datetime") - pl.col("__start")).dt.total_microseconds()

	return (
		frame.with_row_index("row_id")
		.with_columns(
			(pl.col("datetime") - pl.col("time_spent")).alias("__start"), visit.alias("__visit")
		)
		.pipe(split_on_cell_boundaries, recording, "__start", "datetime")
		.with_columns(
			(pl.col("datetime") - pl.col("__start")).alias("time_spent"),
			pl.when(pl.col("__visit") > 0)
			.then(
				pl.duration(
					microseconds=(
						pl.col("time_under")
						.dt.total_microseconds()
						.mul(piece)
						.truediv(pl.col("__visit"))
					)
					.round()
					.cast(pl.Int64)
				)
			)
			.otherwise(pl.col("time_under"))
			.alias("time_under"),
			(~pl.col("first_piece")).alias("interpolated"),
		)
		.drop("__start", "__visit", "first_piece")
	)
