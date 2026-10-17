import datetime as dt
import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations, product
from typing import Literal

import numpy as np
import polars as pl

from deepecohab.core.data_model import Layout
from deepecohab.plotting import durations
from deepecohab.plotting.context import Granularity, PlotContext, Scope

Aggregation = Literal["sum", "mean"]


def prep_animal_speed(
	context: PlotContext,
	days_range: tuple[int, int],
	phase_type: Sequence[str],
	granularity: Granularity = "day",
	hours_range: tuple[int, int] | None = None,
	max_dwell: float = 10,
) -> pl.DataFrame:
	"""Crossing speeds using each tunnel's configured distance and a duration cutoff.

	Retain fractional seconds from the pipeline's Duration column. Both tunnel
	directions count; cage visits and undefined positions never contribute.
	"""
	if not math.isfinite(max_dwell) or max_dwell <= 0:
		raise ValueError("max_dwell must be a positive, finite number of seconds")
	for tunnel in set(context.tunnels_map.values()):
		length = context.tunnel_lengths_cm.get(tunnel)
		if length is None or not math.isfinite(length) or length <= 0:
			raise ValueError(f"Tunnel {tunnel!r} needs a positive, finite crossing length in cm")
	return (
		context.table("main_df")
		.lazy()
		.filter(
			window_filter(days_range, granularity, hours_range),
			pl.col("phase").is_in(phase_type),
			pl.col("position").is_in(list(context.tunnels_map)),
		)
		.with_columns(
			pl.col("time_spent").dt.total_seconds(fractional=True),
			pl.col("position").cast(pl.String).replace_strict(context.tunnels_map),
		)
		.filter(pl.col("time_spent").is_between(0, max_dwell, closed="right"))
		.with_columns(
			(
				pl.col("position").replace_strict(
					context.tunnel_lengths_cm, return_dtype=pl.Float64
				)
				/ pl.col("time_spent")
			).alias("speed_cm_s"),
		)
		.sort("animal_id", "speed_cm_s")
		.collect(engine="in-memory")
	)


def prep_speed_box(
	context: PlotContext,
	days_range: tuple[int, int],
	phase_type: Sequence[str],
	granularity: Granularity,
	hours_range: tuple[int, int] | None = None,
	max_dwell: float = 10,
) -> pl.DataFrame:
	"""One median crossing speed per animal, tunnel and day or phase."""
	return (
		prep_animal_speed(context, days_range, phase_type, granularity, hours_range, max_dwell)
		.group_by("position", "animal_id", granularity)
		.agg(pl.median("speed_cm_s"))
		.sort("position", "animal_id", granularity)
	)


def prep_speed_line(
	context: PlotContext,
	days_range: tuple[int, int],
	phase_type: Sequence[str],
	granularity: Granularity,
	x: Literal["hour", "day", "phase_count"],
	hours_range: tuple[int, int] | None = None,
	max_dwell: float = 10,
) -> pl.DataFrame:
	"""Mean hourly crossing speeds, with SEM over the axis folded away.

	Each observed animal/day-or-phase/hour cell contributes one mean, regardless
	of crossing count. Empty cells remain null and never count as zero speed.
	"""
	frame = prep_animal_speed(context, days_range, phase_type, granularity, hours_range, max_dwell)
	hours = range(24) if hours_range is None else range(hours_range[0], hours_range[1] + 1)
	bins = hours if x == "hour" else range(days_range[0], days_range[1] + 1)
	scaffold = pl.DataFrame(
		product(context.animal_ids, bins),
		schema={"animal_id": frame.schema["animal_id"], x: frame.schema[x]},
		orient="row",
	)
	return (
		frame.group_by("animal_id", granularity, "hour")
		.agg(pl.mean("speed_cm_s"))
		.group_by("animal_id", x)
		.agg(
			pl.mean("speed_cm_s").alias("mean"),
			(pl.std("speed_cm_s") / pl.col("speed_cm_s").count().sqrt()).alias("sem"),
		)
		.with_columns(
			(pl.col("mean") - pl.col("sem")).alias("lower"),
			(pl.col("mean") + pl.col("sem")).alias("upper"),
		)
		.join(scaffold, on=["animal_id", x], how="right")
		.sort("animal_id", x)
	)


@dataclass(frozen=True, eq=False)
class Heatmap:
	"""A faceted matrix and the labels it is drawn with.

	Attributes:
		values: ``(facet, row, column)`` values driving the colour scale.
		text: same shape, pre-formatted for hover, or ``None`` for plain numbers.
		label: colourbar title, including the unit.
		x: column labels.
		y: row labels.
		facets: one title per facet.
	"""

	values: np.ndarray
	text: np.ndarray | None
	label: str
	x: list[int] | list[str]
	y: list[str]
	facets: list[str]


def window_filter(
	days_range: tuple[int, int],
	granularity: Granularity,
	hours_range: tuple[int, int] | None = None,
) -> pl.Expr:
	"""The window predicate every prep function narrows its source table with.

	Args:
		days_range: first and last window unit (day or phase occurrence) to keep.
		granularity: which axis ``days_range`` measures.
		hours_range: first and last hour since the ``start_from`` onset to keep, or
			``None`` to keep every hour. Only tables carrying their own ``hour`` -
			not ``phase_durations``, or anything normalised against it - can take one.
	"""
	expr = pl.col(granularity).is_between(days_range[0], days_range[1])

	if hours_range is not None:
		expr = expr & pl.col("hour").is_between(hours_range[0], hours_range[1])

	return expr


def _matrix(
	frame: pl.DataFrame,
	on: str,
	index: str,
	values: str,
	animals: list[str],
) -> np.ndarray:
	"""Pivot a long pair table into a square animal-by-animal matrix."""
	return frame.pivot(on=on, index=index, values=values).drop(index).select(animals).to_numpy()


def prep_event_spans(
	context: PlotContext,
	days_range: tuple[int, int],
	granularity: Granularity,
	x: Literal["datetime", "hour", "day", "phase_count"],
	hours_range: tuple[int, int] | None = None,
) -> pl.DataFrame:
	"""Where each event falls on a plot's x axis, within the selected window.

	On a binned axis a span covers every bin the event overlaps, from half a bin before
	the first to half a bin after the last, since each point stands for its whole bin.

	Args:
		context: the context whose ``event_bouts`` table is read.
		days_range: first and last window unit to keep.
		granularity: the unit of the window.
		x: the ``event_bouts`` column the plot's x axis shows.
		hours_range: first and last hour to keep, matching the plot it is drawn on.

	Returns:
		One row per span, with ``event``, ``position``, ``x0`` and ``x1``; empty for a
		recording analysed before events existed.
	"""
	if "event_bouts" not in context:
		return pl.DataFrame(schema=["event", "position", "x0", "x1"])

	bouts = context.table("event_bouts").filter(window_filter(days_range, granularity, hours_range))

	# Plotly draws a trace's datetimes as naive wall-clock times, so the spans must be too.
	if x == "datetime":
		return bouts.select(
			"event",
			"position",
			pl.col("start").dt.replace_time_zone(None).alias("x0"),
			pl.col("end").dt.replace_time_zone(None).alias("x1"),
		).unique(maintain_order=True)

	return (
		bouts.select("event", "position", pl.col(x).cast(pl.Int64))
		.unique()
		.sort("event", "position", x)
		# A gap between neighbouring bins starts a new span.
		.with_columns(
			(pl.col(x).diff().over("event", "position") != 1)
			.fill_null(True)
			.cum_sum()
			.over("event", "position")
			.alias("run")
		)
		.group_by("event", "position", "run")
		.agg((pl.col(x).min() - 0.5).alias("x0"), (pl.col(x).max() + 0.5).alias("x1"))
		.drop("run")
	)


def prep_timeline(
	context: PlotContext,
	days_range: tuple[int, int],
	granularity: Granularity,
	hours_range: tuple[int, int] | None = None,
	x_range: tuple[str, str] | None = None,
) -> pl.DataFrame:
	"""Every animal's position intervals within the selected window, as timeline bars.

	Reads ``main_df`` directly rather than a downstream table, so each row is one real
	visit - a registration's ``time_spent`` is the gap since the animal's previous one,
	so it ran from ``datetime - time_spent`` to ``datetime``.

	A week holds hundreds of thousands of registrations, nearly all narrower than a
	pixel, so the span on show - the window, or ``x_range`` once zoomed in - is cut into
	2000 buckets, about one per pixel of a full-width card. Each bucket takes the
	position the animal spent most of it in, undefined included, and neighbouring buckets
	at one position merge into a single bar, trimmed to the defined visits it covers - the
	recording's ends and undefined gaps rarely fall on a bucket edge. Undefined bars are
	then dropped, so a gap wider than a bucket or two always shows. Times, ``x_range``
	included, are naive wall-clock, as plotly draws them.
	"""
	visits = (
		context.table("main_df")
		.lazy()
		.filter(window_filter(days_range, granularity, hours_range))
		.select(
			"animal_id",
			pl.col("position").cast(pl.String).replace(context.tunnels_map),
			(pl.col("datetime") - pl.col("time_spent"))
			.dt.replace_time_zone(None)
			.dt.epoch("ms")
			.alias("start"),
			pl.col("datetime").dt.replace_time_zone(None).dt.epoch("ms").alias("end"),
		)
	)
	defined = pl.col("position") != Layout.UNDEFINED
	if x_range is not None:
		low, high = (pl.lit(dt.datetime.fromisoformat(edge)).dt.epoch("ms") for edge in x_range)
		visits = visits.filter(pl.col("end") > low, pl.col("start") < high).with_columns(
			pl.col("start").clip(low), pl.col("end").clip(upper_bound=high)
		)

	return (
		visits.filter(pl.col("end") > pl.col("start"))
		.with_columns(width=((pl.col("end").max() - pl.col("start").min()) // 2000).clip(1))
		.with_columns(
			bucket=pl.int_ranges(
				pl.col("start") // pl.col("width"), (pl.col("end") - 1) // pl.col("width") + 1
			)
		)
		.explode("bucket", empty_as_null=False)
		.with_columns(
			overlap=pl.min_horizontal("end", (pl.col("bucket") + 1) * pl.col("width"))
			- pl.max_horizontal("start", pl.col("bucket") * pl.col("width"))
		)
		.group_by("animal_id", "bucket", "position")
		.agg(
			pl.col("overlap").sum(),
			pl.col("width").first(),
			pl.col("start").min(),
			pl.col("end").max(),
		)
		.group_by("animal_id", "bucket")
		.agg(
			pl.col("position").sort_by("overlap", "position").last(),
			pl.col("width").first(),
			pl.col("start").filter(defined).min(),
			pl.col("end").filter(defined).max(),
		)
		.sort("animal_id", "bucket")
		.with_columns(
			(
				(pl.col("position") != pl.col("position").shift(1))
				| (pl.col("bucket") != pl.col("bucket").shift(1) + 1)
			)
			.fill_null(True)
			.cum_sum()
			.over("animal_id")
			.alias("run")
		)
		.group_by("animal_id", "position", "run")
		.agg(
			start=pl.max_horizontal(
				pl.col("bucket").min() * pl.col("width").first(), pl.col("start").min()
			),
			end=pl.min_horizontal(
				(pl.col("bucket").max() + 1) * pl.col("width").first(), pl.col("end").max()
			),
		)
		.filter(defined)
		.select("animal_id", "position", pl.from_epoch("start", "ms"), pl.from_epoch("end", "ms"))
		.sort("animal_id", "start")
		.collect(engine="in-memory")
	)


def prep_ranking_over_time(
	context: PlotContext,
	days_range: tuple[int, int],
	granularity: Granularity,
) -> pl.DataFrame:
	"""Aggregate animal ordinal rankings by window unit, hour, and datetime.

	A match moves only its two animals, yet ``ranking`` repeats every animal's rating
	after it. Only the rows where an animal's rating changes are kept, plus its last
	one: the step line holds each rating until the next change anyway.
	"""
	return (
		context.table("ranking")
		.lazy()
		.filter(window_filter(days_range, granularity))
		.sort("datetime")
		.group_by(granularity, "hour", "animal_id", "datetime", maintain_order=True)
		.agg(
			pl.when(pl.first(granularity) == 1)
			.then(pl.first("ordinal"))
			.otherwise(pl.last("ordinal"))
		)
		.filter(
			(pl.col("ordinal").diff().over("animal_id") != 0).fill_null(True)
			| (pl.col("datetime") == pl.col("datetime").max().over("animal_id"))
		)
		.collect(engine="in-memory")
	)


def prep_ranking_day_stability(
	context: PlotContext,
	days_range: tuple[int, int],
	granularity: Granularity,
) -> pl.DataFrame:
	"""Rank animals by their last ordinal in each window unit."""
	return (
		context.table("ranking")
		.lazy()
		.filter(window_filter(days_range, granularity))
		.group_by(granularity, "animal_id")
		.agg(pl.col("ordinal").last())
		.with_columns(
			pl.col("ordinal")
			.rank(method="average", descending=True)
			.over(granularity)
			.alias("rank")
		)
		.sort(granularity, "rank")
		.collect(engine="in-memory")
	)


def prep_ranking_distribution(
	context: PlotContext,
	days_range: tuple[int, int],
	granularity: Granularity,
) -> pl.DataFrame:
	"""Fit a normal probability density over each animal's last ranking in the window.

	The ranking table is in replay order, so an animal's last row is its rating after the
	window's final match - the same whichever granularity the window is given in.
	"""
	pdf_expression = (1 / (pl.col("sigma") * math.sqrt(2 * math.pi))) * (
		-0.5 * ((pl.col("ranking") - pl.col("mu")) / pl.col("sigma")) ** 2
	).exp()

	return (
		context.table("ranking")
		.lazy()
		.filter(window_filter(days_range, granularity))
		.group_by("animal_id")
		.agg(pl.last("mu"), pl.last("sigma"))
		.join(pl.LazyFrame({"ranking": np.arange(-10, 50, 0.1)}), how="cross")
		.with_columns(pdf_expression.alias("probability_density"))
		.select("animal_id", "ranking", "probability_density")
		.sort("animal_id")
		.collect(engine="in-memory")
	)


def prep_polar(
	context: PlotContext,
	days_range: tuple[int, int],
	phase_type: Sequence[str],
	granularity: Granularity,
	hours_range: tuple[int, int] | None = None,
) -> pl.DataFrame:
	"""Z-score every feature metric onto one comparable polar scale."""
	n_bins = days_range[1] - days_range[0] + 1

	return (
		context.table("feature_df")
		.lazy()
		# The whole selection narrows the rows before the z-score, so animals are
		# compared with each other within it.
		.filter(
			pl.col("phase").is_in(list(phase_type)),
			window_filter(days_range, granularity, hours_range),
		)
		.group_by("animal_id", "metric", "phase", "day", "phase_count")
		.agg(pl.sum("value"), pl.sum("exposure"))
		.with_columns(
			pl.when(pl.col("exposure") > 0)
			.then(pl.col("value") / pl.col("exposure"))
			.otherwise(0.0)
			.alias("rate")
		)
		.with_columns(
			pl.when(pl.col("rate").std().fill_null(0) == 0)
			.then(pl.lit(0.0))
			.otherwise((pl.col("rate") - pl.col("rate").mean()) / pl.col("rate").std())
			.over("metric")
			.alias("z-score")
		)
		.group_by("animal_id", "metric", granularity)
		.agg(pl.mean("z-score"))
		.group_by("animal_id", "metric")
		.agg(
			pl.mean("z-score").alias("mean"),
			(pl.std("z-score") / math.sqrt(n_bins)).alias("sem"),
		)
		.with_columns(
			(pl.col("mean") - pl.col("sem")).alias("lower"),
			(pl.col("mean") + pl.col("sem")).alias("upper"),
		)
		.sort("metric", "animal_id")
		.fill_null(0)
		.with_columns(
			pl.col("metric").str.to_titlecase().str.replace_all("_", " ").str.replace("N ", "# of ")
		)
		.collect(engine="streaming")
	)


def prep_network_dominance(
	context: PlotContext,
	days_range: tuple[int, int],
	granularity: Granularity,
) -> tuple[pl.DataFrame, pl.DataFrame]:
	"""Return chasing edges and the latest ranking per animal."""
	animals = context.animal_ids
	join_frame = pl.LazyFrame(
		data=product(animals, animals),
		schema=[("target", pl.Enum(context.animal_ids)), ("source", pl.Enum(context.animal_ids))],
	)

	connections = (
		context.table("chasings_df")
		.lazy()
		.filter(window_filter(days_range, granularity))
		.group_by("chased", "chaser")
		.agg(pl.sum("chasings"))
		.join(join_frame, left_on=["chaser", "chased"], right_on=["source", "target"], how="right")
		.fill_null(0)
		.sort("target", "source")  # necessary for deterministic output
		.collect(engine="in-memory")
	)

	diff = (pl.col(granularity) - days_range[-1]).abs()
	nodes = (
		context.table("ranking")
		.lazy()
		.filter(diff == diff.min())  # last window unit that updated the rank
		.group_by("animal_id")
		.agg(pl.last("ordinal"))
		.collect(engine="in-memory")
	)

	return connections, nodes


def _proportion_together(context: PlotContext, scope: Scope) -> pl.LazyFrame:
	"""Per-pair share of each phase spent together, over the positions in scope.

	``incohort_sociability`` stores this column for cages only, because the chance term it
	is compared against is built from cage occupancy. Every position shares one
	denominator, so summing the per-position ratios and dividing the summed time give the
	same number - which is what lets the scoped value be computed here and still agree
	exactly with the stored one at ``scope="cages"``.
	"""
	phase_seconds = pl.col("duration").dt.total_seconds(fractional=True)

	return (
		context.table("pairwise_meetings")
		.lazy()
		.filter(pl.col("position").is_in(context.scope_positions(scope)))
		.group_by("phase", "day", "phase_count", "animal_id", "animal_id_2")
		.agg(pl.sum("time_together"))
		.join(context.table("phase_durations").lazy(), on=["phase_count", "phase"], how="left")
		.select(
			"phase",
			"day",
			"phase_count",
			"animal_id",
			"animal_id_2",
			(pl.col("time_together").dt.total_seconds(fractional=True) / phase_seconds).alias(
				"proportion_together"
			),
		)
	)


def prep_network_sociability(
	context: PlotContext,
	days_range: tuple[int, int],
	granularity: Granularity,
	scope: Scope = "cages",
) -> pl.DataFrame:
	"""Return edges weighted by the time each pair spent together."""
	animals = context.animal_ids
	join_frame = pl.LazyFrame(
		data=combinations(animals, 2),
		schema=[("source", pl.Enum(context.animal_ids)), ("target", pl.Enum(context.animal_ids))],
	)

	return (
		_proportion_together(context, scope)
		.filter(window_filter(days_range, granularity))
		.group_by("animal_id", "animal_id_2")
		.agg(pl.sum("proportion_together"))
		.join(
			join_frame,
			left_on=["animal_id", "animal_id_2"],
			right_on=["source", "target"],
			how="right",
		)
		.sort("source", "target")  # order fixes the node positions
		.fill_null(0)
		.collect(engine="in-memory")
	)


def prep_directed_heatmap(
	context: PlotContext,
	days_range: tuple[int, int],
	phase_type: Sequence[str],
	agg: Aggregation,
	granularity: Granularity,
	animals: list[str],
	table: str,
	value: str,
	column: str,
	row: str,
	hours_range: tuple[int, int] | None = None,
) -> np.ndarray:
	"""Pivot a directed pair count into a ``column``-versus-``row`` matrix.

	Args:
		table: the pair table to read, such as ``chasings_df``.
		value: the count column in ``table``, such as ``chasings``.
		column: the animal column laid along the matrix columns, such as ``chaser``.
		row: the animal column laid along the matrix rows, such as ``chased``.
	"""
	join_frame = pl.LazyFrame(
		product(animals, animals),
		schema=[(row, pl.Enum(context.animal_ids)), (column, pl.Enum(context.animal_ids))],
	)

	match agg:
		case "sum":
			agg_func = pl.sum(value).alias("sum")
		case "mean":
			agg_func = pl.mean(value).round(2).alias("mean")

	frame = (
		context.table(table)
		.lazy()
		.sort(row, column)
		.filter(
			pl.col("phase").is_in(list(phase_type)),
			window_filter(days_range, granularity, hours_range),
		)
		.group_by(granularity, column, row)
		.agg(pl.sum(value))
		.group_by(column, row, maintain_order=True)
		.agg(agg_func)
		.join(join_frame, on=[column, row], how="right", maintain_order="right")
		.collect(engine="in-memory")
	)

	return _matrix(frame, on=column, index=row, values=agg, animals=animals)


def prep_count_line(
	context: PlotContext,
	days_range: tuple[int, int],
	granularity: Granularity,
	x: Literal["hour", "day", "phase_count"],
	table: str,
	animal_column: str,
	count: pl.Expr,
	hours_range: tuple[int, int] | None = None,
) -> pl.DataFrame:
	"""Totals per animal along ``x``, with the mean and SEM over the axis it folds away.

	Counts are taken per animal, window unit and hour, then collapsed onto ``x``: along
	``"hour"`` each hour of the day gathers its window units, along the granularity each
	window unit gathers its hours.

	Args:
		x: the axis to keep - ``"hour"``, or the granularity for its window units.
		table: the table to read, such as ``main_df``.
		animal_column: the animal column to total by, such as ``chaser``.
		count: what the rows of one animal, hour and window unit add up to, such as
			``pl.len()`` for antenna detections.
		hours_range: first and last hour to keep; the zero-fill scaffold is narrowed
			to match, so a dropped hour is absent rather than a zero bar.
	"""
	hours = range(24) if hours_range is None else range(hours_range[0], hours_range[1] + 1)

	join_frame = pl.LazyFrame(
		product(context.animal_ids, range(days_range[0], days_range[1] + 1), hours),
		schema=[
			(animal_column, pl.Enum(context.animal_ids)),
			(granularity, pl.Int16()),
			("hour", pl.Int8()),
		],
	)

	return (
		context.table(table)
		.lazy()
		.filter(window_filter(days_range, granularity, hours_range))
		.group_by(granularity, "hour", animal_column)
		.agg(count.alias("count"))
		.join(join_frame, on=[animal_column, "hour", granularity], how="right")
		.fill_null(0)
		.group_by(x, animal_column)
		.agg(
			pl.sum("count").alias("total"),
			pl.mean("count").alias("mean").round(2),
			(pl.std("count") / pl.len().sqrt()).alias("sem"),
		)
		.with_columns(
			(pl.col("mean") - pl.col("sem")).alias("lower"),
			(pl.col("mean") + pl.col("sem")).alias("upper"),
		)
		.sort(animal_column, x)
		.collect(engine="in-memory")
	)


def prep_activity(
	context: PlotContext,
	days_range: tuple[int, int],
	phase_type: Sequence[str],
	granularity: Granularity,
	agg: Aggregation,
	positions: list[str],
	columns: dict[str, str],
	hours_range: tuple[int, int] | None = None,
) -> pl.DataFrame:
	"""``activity_df`` columns summed per position and animal, keyed ``{output: source}``.

	A summed plot draws one row per animal and position; a mean plot keeps the
	window units so the box has a distribution to show. Aggregating here rather
	than in the plot is what keeps the hover text describing the value on screen -
	the catalog step still owns turning Durations into a display unit, once any
	group averaging has run.
	"""
	per_unit = (
		context.table("activity_df")
		.lazy()
		.with_columns(pl.col("position").cast(pl.String))
		.filter(
			pl.col("phase").is_in(list(phase_type)),
			window_filter(days_range, granularity, hours_range),
			pl.col("position").is_in(positions),
		)
		.group_by(granularity, "animal_id", "position")
		.agg(pl.sum(source).alias(output) for output, source in columns.items())
	)

	if agg == "sum":
		per_unit = per_unit.group_by("animal_id", "position").agg(pl.sum(*columns))

	return per_unit.sort("animal_id", "position").collect(engine="in-memory")


def _facet_matrices(
	frame: pl.DataFrame,
	facet: str,
	row: str,
	column: str,
	values: str,
	facets: list[str],
	rows: list[str],
	columns: list,
) -> np.ndarray:
	"""Pivot a long table into one ``(row, column)`` matrix per facet."""
	wide = (
		frame.pivot(on=column, index=[facet, row], values=values)
		.with_columns(pl.col(facet).cast(pl.String), pl.col(row).cast(pl.String))
		.sort(
			pl.col(facet).replace_strict(facets, range(len(facets)), return_dtype=pl.Int32),
			pl.col(row).replace_strict(rows, range(len(rows)), return_dtype=pl.Int32),
		)
		.select([str(label) for label in columns])
	)

	return wide.to_numpy().reshape(len(facets), len(rows), len(columns))


def prep_time_per_position(
	context: PlotContext,
	days_range: tuple[int, int],
	agg: Aggregation,
	granularity: Granularity,
	x: Literal["hour", "day", "phase_count"],
	positions: list[str],
	animals: list[str],
	unit: durations.Unit | Literal["auto"] = "auto",
	hours_range: tuple[int, int] | None = None,
) -> Heatmap:
	"""Time spent per position, as one animal-by-``x`` matrix per position.

	Args:
		x: the matrix columns - ``"hour"`` for the hours of the day, or the
			granularity for its window units. When it is ``"hour"``, ``hours_range``
			narrows the columns themselves rather than just which rows contribute.
	"""
	if x == "hour":
		x_values = list(
			range(24) if hours_range is None else range(hours_range[0], hours_range[1] + 1)
		)
	else:
		x_values = list(range(days_range[0], days_range[1] + 1))

	join_frame = pl.LazyFrame(
		product(x_values, positions, animals),
		schema=[
			(x, pl.Int16()),
			("position", pl.String()),
			("animal_id", pl.Enum(context.animal_ids)),
		],
	)

	match agg:
		case "sum":
			agg_func = pl.sum("time_in_position")
		case "mean":
			agg_func = pl.mean("time_in_position")

	frame = (
		context.table("activity_df")
		.lazy()
		.with_columns(pl.col("position").cast(pl.String))
		.filter(
			window_filter(days_range, granularity, hours_range),
			pl.col("position").is_in(positions),
		)
		.group_by([x, "animal_id", "position"])
		.agg(agg_func)
		.join(join_frame, on=[x, "position", "animal_id"], how="right")
		.collect(engine="in-memory")
	)

	frame, label = durations.to_display(frame, "time_in_position", unit, "Time spent")
	shape = ("position", "animal_id", x)

	return Heatmap(
		values=_facet_matrices(frame, *shape, "time_in_position", positions, animals, x_values),
		text=_facet_matrices(frame, *shape, "time_in_position_text", positions, animals, x_values),
		label=label,
		x=x_values,
		y=animals,
		facets=positions,
	)


def prep_cage_preference(
	context: PlotContext,
	days_range: tuple[int, int],
	phase_type: Sequence[str],
	granularity: Granularity,
	positions: list[str],
	unit: durations.Unit | Literal["auto"] = "auto",
	hours_range: tuple[int, int] | None = None,
) -> tuple[pl.DataFrame, str]:
	"""Time spent per position, per animal and window unit."""
	frame = (
		context.table("activity_df")
		.lazy()
		.filter(
			pl.col("phase").is_in(list(phase_type)),
			window_filter(days_range, granularity, hours_range),
		)
		.with_columns(pl.col("position").cast(pl.String))
		.group_by(granularity, "animal_id", "position")
		.agg(pl.sum("time_in_position"))
		.filter(pl.col("position").is_in(positions))
		.sort("position")
		.collect(engine="in-memory")
	)

	return durations.to_display(frame, "time_in_position", unit, "Time spent")


def prep_pairwise_sociability(
	context: PlotContext,
	days_range: tuple[int, int],
	phase_type: Sequence[str],
	agg: Aggregation,
	metric: Literal["time_together", "pairwise_encounters"],
	granularity: Granularity,
	positions: list[str],
	animals: list[str],
	unit: durations.Unit | Literal["auto"] = "auto",
	hours_range: tuple[int, int] | None = None,
) -> Heatmap:
	"""Pivot pairwise meetings into one animal-by-animal matrix per position."""
	join_frame = pl.LazyFrame(
		product(positions, animals, animals),
		schema=[
			("position", pl.Categorical()),
			("animal_id", pl.Enum(context.animal_ids)),
			("animal_id_2", pl.Enum(context.animal_ids)),
		],
	)

	frame = (
		context.table("pairwise_meetings")
		.lazy()
		.filter(
			pl.col("phase").is_in(list(phase_type)),
			window_filter(days_range, granularity, hours_range),
		)
		.group_by(["animal_id", "animal_id_2", "position"], maintain_order=True)
		.agg(
			pl.sum(metric).alias("sum"),
			pl.mean(metric).alias("mean"),
		)
		.join(join_frame, on=["position", "animal_id", "animal_id_2"], how="right")
		.collect(engine="in-memory")
	)

	shape = ("position", "animal_id", "animal_id_2")

	if metric == "pairwise_encounters":
		return Heatmap(
			values=_facet_matrices(frame, *shape, agg, positions, animals, animals),
			text=None,
			label="<b>Number</b>",
			x=animals,
			y=animals,
			facets=positions,
		)

	frame, label = durations.to_display(frame, agg, unit, "Time together")

	return Heatmap(
		values=_facet_matrices(frame, *shape, agg, positions, animals, animals),
		text=_facet_matrices(frame, *shape, f"{agg}_text", positions, animals, animals),
		label=label,
		x=animals,
		y=animals,
		facets=positions,
	)


def prep_within_cohort_sociability(
	context: PlotContext,
	days_range: tuple[int, int],
	phase_type: Sequence[str],
	metric: Literal["proportion_together", "sociability"],
	granularity: Granularity,
	animals: list[str],
	scope: Scope = "cages",
) -> np.ndarray:
	"""Pivot mean cohort sociability into an animal-by-animal matrix.

	``scope`` applies to ``proportion_together`` only. ``sociability`` is measured against
	a cage-only chance expectation, so it is read from ``incohort_sociability`` as stored.
	"""
	join_frame = pl.LazyFrame(
		product(animals, animals),
		schema=[
			("animal_id", pl.Enum(context.animal_ids)),
			("animal_id_2", pl.Enum(context.animal_ids)),
		],
	)

	source = (
		_proportion_together(context, scope)
		if metric == "proportion_together"
		else context.table("incohort_sociability").lazy()
	)

	frame = (
		source.with_columns(pl.col(metric).round(3))
		.filter(
			pl.col("phase").is_in(list(phase_type)),
			window_filter(days_range, granularity),
		)
		.group_by(["animal_id", "animal_id_2"], maintain_order=True)
		.agg(pl.mean(metric).round(2).alias("mean"))
		.join(join_frame, on=["animal_id", "animal_id_2"], how="right", maintain_order="right")
		.collect(engine="in-memory")
	)

	return _matrix(frame, on="animal_id_2", index="animal_id", values="mean", animals=animals)


def prep_social_stability(
	context: PlotContext,
	days_range: tuple[int, int],
	phase_type: Sequence[str],
	granularity: Granularity,
	scope: Scope = "cages",
) -> pl.DataFrame:
	"""Median time together per pair, with a stability score from its dispersion."""
	mad = (pl.col("proportion_together") - pl.median("proportion_together")).abs().median()

	frame = _proportion_together(context, scope)
	frame = pl.concat(
		[frame, frame.rename({"animal_id": "animal_id_2", "animal_id_2": "animal_id"})]
	)

	return (
		frame.filter(
			pl.col("phase").is_in(list(phase_type)),
			window_filter(days_range, granularity),
		)
		.group_by(granularity, "animal_id", "animal_id_2")
		.agg(pl.mean("proportion_together"))
		.sort("animal_id", "animal_id_2", granularity)
		.group_by("animal_id", "animal_id_2")
		.agg(
			# +1e-10 avoids a division by zero turning stability into NaN
			(1 - (mad / (pl.median("proportion_together") + 1e-10)))
			.clip(0, 1)
			.round(2)
			.alias("stability"),
			pl.median("proportion_together").round(2),
		)
		.sort("animal_id")
		.collect(engine="in-memory")
	)


def prep_quality_heatmap(context: PlotContext, animals: list[str]) -> tuple[np.ndarray, list[str]]:
	"""Miss rate per animal and antenna, as an animal-by-antenna matrix.

	``recording_quality`` already has one row per animal and antenna - it is built
	from a cross join - so no zero-filling scaffold is needed here, unlike the pair
	matrices.

	Returns:
		The matrix, its rows in ``animals`` order, and the antenna column labels.
	"""
	frame = context.table("recording_quality")
	antennas = [str(antenna) for antenna in sorted(frame["antenna"].unique().to_list())]

	wide = (
		frame.pivot(on="antenna", index="animal_id", values="miss_rate")
		.with_columns(pl.col("animal_id").cast(pl.String))
		.sort(
			pl.col("animal_id").replace_strict(animals, range(len(animals)), return_dtype=pl.Int32)
		)
		.select(antennas)
	)

	return wide.to_numpy(), antennas


#: The kinds of pass ``recording_quality`` counts, in stacking order.
READ_KINDS = ("correct", "interpolated", "bad")


def prep_quality_by_antenna(context: PlotContext) -> pl.DataFrame:
	"""Passes per antenna by kind, pooled across the cohort, each as a share of all passes.

	Pooled shares are ``kind.sum() / passes.sum()``, never the mean of the per-animal
	cell shares, so a lightly-sampled animal cannot skew an antenna as much as a
	heavily-sampled one.

	Returns:
		One row per antenna: the ``READ_KINDS`` counts and a ``<kind>_share`` percentage
		of each.
	"""
	passes = pl.sum_horizontal(READ_KINDS)

	return (
		context.table("recording_quality")
		.lazy()
		.group_by("antenna")
		.agg(pl.col(READ_KINDS).sum())
		.with_columns(
			pl.when(passes > 0)
			.then(100 * pl.col(kind) / passes)
			.otherwise(0.0)
			.alias(f"{kind}_share")
			for kind in READ_KINDS
		)
		.sort("antenna")
		.collect(engine="in-memory")
	)


def prep_actogram(
	context: PlotContext,
	days_range: tuple[int, int],
	granularity: Granularity,
	hours_range: tuple[int, int] | None = None,
) -> tuple[Heatmap, pl.DataFrame]:
	"""Cohort visits on a day-by-hour-of-day grid, and the cells an event covers.

	Rows are experiment days whatever the window's granularity: a phase is half a day,
	so a phase axis here would leave every other row empty. A phase window narrows which
	hours of those days are kept instead.

	An hour nothing was recorded in stays null rather than zero - a day the recording
	stopped part way through then reads as blank cells, not as a quiet stretch.

	Returns:
		The grid as a single-facet :class:`Heatmap`, and one row per ``(event, day,
		hour)`` cell an event bout touches - empty for a recording analysed before
		events existed.
	"""
	frame = (
		context.table("activity_df")
		.lazy()
		.filter(
			window_filter(days_range, granularity, hours_range),
			pl.col("position").cast(pl.String) != Layout.UNDEFINED,
		)
		.group_by("day", "hour")
		.agg(pl.sum("visits_to_position").alias("visits"))
		.collect(engine="in-memory")
	)
	days = sorted(frame["day"].unique().to_list())
	hours = sorted(frame["hour"].unique().to_list()) or list(range(24))
	grid = {(row["day"], row["hour"]): row["visits"] for row in frame.iter_rows(named=True)}
	values = np.array(
		[[[grid.get((day, hour), np.nan) for hour in hours] for day in days]], dtype=float
	)

	schema = {"event": pl.String, "index": pl.Int64, "day": pl.Int64, "hour": pl.Int64}
	cells = pl.DataFrame(schema=schema)
	if "event_bouts" in context:
		cells = (
			context.table("event_bouts")
			.filter(window_filter(days_range, granularity, hours_range))
			.select(
				pl.col("event").cast(pl.String),
				# An Enum's physical code is the event's place among the recording's
				# events, which is how _event_spans picks its colour too.
				pl.col("event").to_physical().cast(pl.Int64).alias("index"),
				pl.col("day", "hour").cast(pl.Int64),
			)
			.unique()
		)

	return (
		Heatmap(
			values=values,
			text=None,
			label="<b>Visits</b>",
			x=hours,
			y=[f"Day {day}" for day in days],
			facets=["visits"],
		),
		cells,
	)


def prep_occupancy_share(
	context: PlotContext,
	days_range: tuple[int, int],
	granularity: Granularity,
	hours_range: tuple[int, int] | None = None,
) -> pl.DataFrame:
	"""Share of the cohort's time each place held, per window unit.

	Tunnels collapse into one ``tunnels`` band - transit, not habitat, and a tunnel holds
	a fraction of a percent on its own. ``undefined`` keeps a band of its own: it is not a
	place, but the share of the record that could not be placed belongs on the same axis
	as the places.

	Returns:
		``granularity``, ``place``, ``phase`` and ``pct``, summing to 100 per window unit.
		``phase`` names the bin's phase on the phase axis, and is arbitrary on the day one.
	"""
	place = (
		pl.when(pl.col("position").cast(pl.String).is_in(context.tunnels))
		.then(pl.lit("tunnels"))
		.otherwise(pl.col("position").cast(pl.String))
		.alias("place")
	)

	return (
		context.table("activity_df")
		.lazy()
		.filter(window_filter(days_range, granularity, hours_range))
		.with_columns(place)
		.group_by(granularity, "place")
		# One row per place per bin, so the stack has one point per x. A phase_count bin
		# is one phase by construction, so first() names it; a day holds both and the
		# caller only reads this column on the phase axis.
		.agg(
			pl.col("phase").first(),
			pl.col("time_in_position").sum().dt.total_seconds().alias("seconds"),
		)
		.with_columns(
			(100 * pl.col("seconds") / pl.col("seconds").sum().over(granularity)).alias("pct")
		)
		.sort(granularity)
		.collect(engine="in-memory")
	)


def prep_phenotype(
	context: PlotContext,
	days_range: tuple[int, int],
	granularity: Granularity,
	hours_range: tuple[int, int] | None = None,
) -> pl.DataFrame:
	"""One row per animal: locomotion, sociality, chases won and dominance rating.

	Sociality is ``1 - time_alone / time_in_position`` over the cages - a ratio of the
	summed durations, never the mean of the per-hour ratios, which would weight a minute
	in a cage the same as an hour. It is null for an animal with no cage time at all.

	Returns:
		``animal_id``, ``subject_name``, ``visits``, ``gregariousness``, ``won`` and
		``ordinal``, ordered by rating.
	"""
	tag = pl.col("animal_id").cast(pl.String)
	activity = (
		context.table("activity_df")
		.lazy()
		.filter(window_filter(days_range, granularity, hours_range))
		.with_columns(tag, pl.col("position").cast(pl.String))
	)
	in_cage, alone = pl.col("in_cage"), pl.col("alone")

	social = (
		activity.filter(pl.col("position").is_in(context.cages))
		.group_by("animal_id")
		.agg(
			pl.col("time_in_position").sum().dt.total_seconds().alias("in_cage"),
			pl.col("time_alone").sum().dt.total_seconds().alias("alone"),
		)
		.with_columns(
			pl.when(in_cage > 0).then(1 - alone / in_cage).alias("gregariousness"),
		)
	)
	visits = (
		activity.filter(pl.col("position") != Layout.UNDEFINED)
		.group_by("animal_id")
		.agg(pl.sum("visits_to_position").alias("visits"))
	)
	won = (
		context.table("chasings_df")
		.lazy()
		.filter(window_filter(days_range, granularity, hours_range))
		.group_by("chaser")
		.agg(pl.sum("chasings").alias("won"))
		.select(pl.col("chaser").cast(pl.String).alias("animal_id"), "won")
	)
	rating = (
		context.table("ranking")
		.lazy()
		.sort("datetime")
		.group_by("animal_id")
		.last()
		.select(tag, "ordinal")
	)

	return (
		social.join(visits, on="animal_id", how="left")
		.join(won, on="animal_id", how="left")
		.join(rating, on="animal_id", how="left")
		.join(context.animals.lazy().select(tag, "subject_name"), on="animal_id", how="left")
		.with_columns(pl.col("visits", "won").fill_null(0))
		.sort("ordinal", nulls_last=True)
		.collect(engine="in-memory")
	)
