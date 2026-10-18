import inspect
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field as dc_field
from typing import Any

import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import polars.selectors as cs

from deepecohab.app.builder.catalog import (
	AGE,
	AGE_UNITS,
	EXPOSURE,
	METRIC,
	VALUE,
	Field,
	Kind,
	metric_names,
)
from deepecohab.plotting import theme

#: Builder figures render with this template initially; the app callback overrides it
#: to match the viewer's theme before the figure reaches the browser.
TEMPLATE = "dark"

#: How a metric measure collapses. The rate is the only one that reads the same at
#: every grouping level, so it leads and is the default.
MEASURE_MODES: dict[str, str] = {
	"rate": "rate",
	"total": "total",
	"exposure": "exposure, h",
	"mean": "hourly mean",
}
DEFAULT_MODE = "rate"

#: Every plotly express encoding we are willing to expose, in shelf order.
CHANNELS: tuple[str, ...] = (
	"x",
	"y",
	"z",
	"r",
	"theta",
	"path",
	"values",
	"color",
	"symbol",
	"size",
	"line_dash",
	"line_group",
	"pattern_shape",
	"facet_col",
	"facet_row",
	"animation_frame",
	"text",
	"hover_data",
)

#: Hover only describes a point, so a category there is listed, never grouped by.
HOVER = "hover_data"

#: Past this many values a hover lists only how many there are.
HOVER_LIST_MAX = 3

#: Not a plotly argument: fields here only add keys to the group-by, which is how a box
#: gets one point per animal instead of one per hour.
DETAIL = "detail"

#: Not a plotly argument either: a box or violin trace takes one marker colour, so the
#: points of this field are redrawn as extra point-only traces, one per colour.
POINT_COLOR = "point_color"

#: Plot types whose traces draw their own points, so Point colour has something to colour.
POINTED: frozenset[str] = frozenset({"box", "violin"})

#: How far past its box's quartiles a point may sit, in interquartile ranges, before
#: excluding outliers drops it: Tukey's fences, where plotly stops a box's whiskers.
FENCE = 1.5

#: How many steps a measure on Point colour is cut into along the colour scale.
POINT_BANDS = 10

#: The column that ties each drawn point back to its row of the aggregated frame.
ROW = "__row"

#: Points whose Point colour measure is empty, in the muted grey the placeholder text uses.
MISSING = "rgb(127, 140, 165)"

#: Plot types that draw a spread rather than a point, so they are meaningless until
#: something on Detail says what one observation is.
DISTRIBUTIONS: frozenset[str] = frozenset({"box", "violin", "strip", "histogram", "ecdf"})

#: Plot types plotly draws as histograms, pooling rows per bin rather than drawing them.
BINNED: frozenset[str] = frozenset({"histogram", "heatmap"})

#: The finest identity a project table holds: one animal within one recording. Tags are
#: reused between recordings, so animal_id on its own would merge different animals.
OBSERVATION: tuple[str, ...] = ("recording", "animal_id")

#: Which chips each shelf accepts. Anything absent takes every kind.
ACCEPTS: dict[str, frozenset[Kind]] = {
	"z": frozenset({"measure"}),
	"size": frozenset({"measure"}),
	"values": frozenset({"measure"}),
	"path": frozenset({"dimension", "time"}),
	"symbol": frozenset({"dimension", "time"}),
	"line_dash": frozenset({"dimension", "time"}),
	"line_group": frozenset({"dimension", "time"}),
	"pattern_shape": frozenset({"dimension", "time"}),
	"facet_col": frozenset({"dimension", "time"}),
	"facet_row": frozenset({"dimension", "time"}),
	"animation_frame": frozenset({"dimension", "time"}),
	DETAIL: frozenset({"dimension", "time"}),
}

#: Shelves that hold an ordered list instead of replacing on every drop.
MULTI: frozenset[str] = frozenset({"path", HOVER, DETAIL})

#: What the Blocks box on an ordered field's chip accepts, and the one example it shows.
BIN_HINT = "3, or 1-3, 4-6"

LABELS: dict[str, str] = {
	"x": "X",
	"y": "Y",
	"z": "Z",
	"r": "Radius",
	"theta": "Angle",
	"path": "Path",
	"values": "Values",
	"color": "Colour",
	"symbol": "Marker",
	"size": "Size",
	"line_dash": "Dash",
	"line_group": "Line group",
	"pattern_shape": "Pattern",
	"facet_col": "Facet col",
	"facet_row": "Facet row",
	"animation_frame": "Animate",
	"text": "Label",
	HOVER: "Hover",
	POINT_COLOR: "Point colour",
	DETAIL: "Detail",
}


@dataclass(frozen=True)
class PlotType:
	"""One entry in the plot picker.

	Attributes:
		name: picker key, also what the builder state holds.
		label: what the picker reads as.
		builder: the plotly express function to call.
		required: channels that must be filled before a figure can be drawn.
		extra: keyword arguments always passed to ``builder``.
	"""

	name: str
	label: str
	builder: Any
	required: tuple[str, ...] = ()
	extra: dict[str, Any] = dc_field(default_factory=dict)

	@property
	def channels(self) -> tuple[str, ...]:
		"""The shelves this plot type offers, read off its own signature."""
		accepted = inspect.signature(self.builder).parameters
		# A bin pools many rows, which leaves a hover field nothing to attach to.
		skipped = HOVER if self.name in BINNED else None
		pointed = (POINT_COLOR,) if self.name in POINTED else ()
		return (*(c for c in CHANNELS if c in accepted and c != skipped), *pointed, DETAIL)


#: Per plot type, the one plotly argument the toolbar lets the user pick and its choices as
#: value to button label. The first choice is the default, which is also what a state with
#: no entry for the argument draws.
OPTIONS: dict[str, tuple[str, dict[str, str]]] = {
	"bar": (
		"barmode",
		{"group": "Grouped", "stack": "Stacked", "relative": "Relative", "overlay": "Overlaid"},
	),
	"box": ("boxmode", {"group": "Grouped", "overlay": "Overlaid"}),
	"line": (
		"line_shape",
		{
			"linear": "Straight",
			"spline": "Smooth",
			"hv": "Step hv",
			"vh": "Step vh",
			"hvh": "Step hvh",
		},
	),
}
OPTIONS["area"] = OPTIONS["line"]

PLOTS: tuple[PlotType, ...] = (
	PlotType("line", "Line", px.line, ("x", "y")),
	PlotType("scatter", "Scatter", px.scatter, ("x", "y")),
	PlotType("bar", "Bar", px.bar, ("x", "y")),
	PlotType("area", "Area", px.area, ("x", "y")),
	PlotType("box", "Box", px.box, ("y",), {"points": "all"}),
	PlotType("violin", "Violin", px.violin, ("y",), {"box": False, "points": "all"}),
	PlotType("strip", "Strip", px.strip, ("y",)),
	PlotType("histogram", "Histogram", px.histogram, ("x",)),
	PlotType("ecdf", "ECDF", px.ecdf, ("x",)),
	# One cell per X and Y value (see build_figure), so avg only pools the Detail groups a
	# cell holds; plotly's default sum would add their rates up.
	PlotType("heatmap", "Heatmap", px.density_heatmap, ("x", "y", "z"), {"histfunc": "avg"}),
	PlotType("scatter_polar", "Polar scatter", px.scatter_polar, ("r", "theta")),
	PlotType("line_polar", "Polar line", px.line_polar, ("r", "theta"), {"line_close": True}),
	PlotType("bar_polar", "Polar bar", px.bar_polar, ("r", "theta")),
	PlotType("sunburst", "Sunburst", px.sunburst, ("path", "values")),
	PlotType("treemap", "Treemap", px.treemap, ("path", "values")),
)

BY_NAME: dict[str, PlotType] = {plot.name: plot for plot in PLOTS}


def plot_type(name: str) -> PlotType:
	"""The plot type called ``name``, falling back to the first one."""
	return BY_NAME.get(name, PLOTS[0])


def accepts(channel: str, kind: Kind) -> bool:
	"""Whether a chip of ``kind`` may be dropped on ``channel``."""
	return kind in ACCEPTS.get(channel, frozenset({"dimension", "time", "measure"}))


def new_state(name: str = PLOTS[0].name, mode: str = DEFAULT_MODE) -> dict[str, Any]:
	"""An empty builder state for the given plot type."""
	return {"kind": name, "channels": {}, "filters": {}, "measure_as": mode}


def seed_detail(state: dict[str, Any], catalog: Sequence[Field]) -> dict[str, Any]:
	"""Give a distribution something to be a distribution over.

	A box whose Detail shelf is empty is one aggregated number, so it draws a single
	point. Filling the shelf with the identity columns - visibly, as chips the user can
	take off again - makes each box a spread over animals instead. Only for a plot type
	that draws a spread, and only while Detail is still empty.
	"""
	if state["kind"] not in DISTRIBUTIONS or state["channels"].get(DETAIL):
		return state

	available = {item.name for item in catalog}
	seeded = [name for name in OBSERVATION if name in available]

	if seeded:
		state["channels"][DETAIL] = seeded

	return state


def label_for(field: Field, mode: str) -> str:
	"""What a field's axis reads as, which for Value depends on how it collapses."""
	if field.agg != "metric":
		return field.label

	return f"{field.label} ({MEASURE_MODES.get(mode, mode)})"


def assigned(state: dict[str, Any]) -> dict[str, list[str]]:
	"""The channels that currently hold at least one field."""
	return {channel: names for channel, names in state["channels"].items() if names}


def _shelved(
	state: dict[str, Any], catalog: Sequence[Field], *, discrete: bool, skip: str = ""
) -> list[Field]:
	"""Fields of one kind on a shelf other than ``skip``, in shelf order and without repeats."""
	fields = {item.name: item for item in catalog}
	picked: dict[str, Field] = {}

	for channel, names in assigned(state).items():
		if channel == skip:
			continue
		for name in names:
			item = fields.get(name)
			if item is not None and item.discrete is discrete:
				picked.setdefault(name, item)

	return list(picked.values())


def group_keys(state: dict[str, Any], catalog: Sequence[Field]) -> list[str]:
	"""Every discrete field on a shelf, in shelf order and without repeats.

	These are what the frame is grouped by, so the measure is computed once per drawn
	point rather than once per hourly row. Hover is left out: see :func:`hover_listed`.
	"""
	return [item.name for item in _shelved(state, catalog, discrete=True, skip=HOVER)]


def hover_listed(state: dict[str, Any], catalog: Sequence[Field]) -> list[str]:
	"""Discrete fields only on Hover: listed per point rather than grouped by.

	Grouping by them would split every point into one per value, so adding a field to
	the hover would change what is drawn.
	"""
	keys = set(group_keys(state, catalog))
	discrete = {item.name for item in catalog if item.discrete}
	return [
		name
		for name in dict.fromkeys(state["channels"].get(HOVER, []))
		if name in discrete and name not in keys
	]


def _listing(name: str, column: pl.Expr) -> pl.Expr:
	"""A point's distinct values of ``column``, or only how many once they are too many."""
	count = column.n_unique()
	return (
		pl.when(count <= HOVER_LIST_MAX)
		.then(column.unique().sort().cast(pl.String).str.join(", "))
		.otherwise(pl.format("{} values", count))
		.alias(name)
	)


def measures(state: dict[str, Any], catalog: Sequence[Field]) -> list[Field]:
	"""Every aggregating field on a shelf, without repeats: what :func:`group_keys` groups."""
	return _shelved(state, catalog, discrete=False)


def apply_filters(frame: pl.LazyFrame, filters: dict[str, Any]) -> pl.LazyFrame:
	"""Narrow the frame to what the filter rail selects.

	Args:
		filters: column to a list of accepted values, or to a ``lo``/``hi`` range.

	Returns:
		The filtered frame; a filter with nothing selected is ignored.
	"""
	schema = frame.collect_schema()

	for column, chosen in filters.items():
		if column not in schema or not chosen:
			continue

		if isinstance(chosen, dict):
			frame = frame.filter(pl.col(column).is_between(chosen["lo"], chosen["hi"]))
		else:
			frame = frame.filter(pl.col(column).cast(pl.String).is_in(list(chosen)))

	return frame


#: One field's blocks: an even block size, or the exact spans the user typed.
Bins = int | list[tuple[float, float]]


def parse_bins(spec: str) -> Bins | None:
	"""Read what the Blocks box holds, or ``None`` when it holds nothing usable.

	Two shapes, told apart by the dash - every ordered field the builder offers counts
	up from zero, so a leading minus is never a value:

	* ``3`` - even blocks of three, however many the data runs to.
	* ``1-3, 4-6, 7-14`` - exactly those spans, in that order, any width, gaps allowed.

	Both ends of a span are included, so ``1-3`` is three days and ``1-3, 4-6`` tiles
	without gap or overlap.

	A value in none of the typed spans is dropped; a value in two lands in the first.
	"""
	text = str(spec).strip()

	if "-" not in text:
		return size if text.isdigit() and (size := int(text)) > 1 else None

	ranges = []
	for part in text.split(","):
		low, _, high = part.partition("-")
		try:
			bounds = (float(low), float(high))
		except ValueError:
			return None
		if bounds[1] < bounds[0]:
			return None
		ranges.append(bounds)

	return ranges or None


def apply_bins(frame: pl.LazyFrame, bins: dict[str, Bins]) -> pl.LazyFrame:
	"""Replace each binned column with the integer key of the block its value falls in.

	The key is the block's own start for an even size, or the span's position in the
	typed order; either way it stays numeric, so grouping and sorting still run in
	order. What span it stands for is only spelled out once the frame is aggregated.
	A value outside every typed span belongs to no block, so its rows are dropped.
	"""
	for name, spec in bins.items():
		if isinstance(spec, int):
			column = pl.col(name)
			frame = frame.with_columns(
				((column - column.min()) // spec * spec + column.min()).alias(name)
			)
			continue

		key = pl.lit(None, pl.Int32)
		for index, (low, high) in reversed(list(enumerate(spec))):
			key = pl.when(pl.col(name).is_between(low, high)).then(index).otherwise(key)
		frame = frame.with_columns(key.alias(name)).filter(pl.col(name).is_not_null())

	return frame


def label_bins(data: pl.DataFrame, bins: dict[str, Bins], catalog: Sequence[Field]) -> pl.DataFrame:
	"""Rewrite each block key as the span it covers, e.g. ``day 4-6``."""
	labels = {item.name: item.label for item in catalog}
	spans = []

	for name, spec in bins.items():
		if name not in data.columns:
			continue
		pretty = labels.get(name, name)
		if isinstance(spec, int):
			span = pl.format("{} {}-{}", pl.lit(pretty), pl.col(name), pl.col(name) + spec - 1)
		else:
			span = pl.col(name).replace_strict(
				list(range(len(spec))),
				[f"{pretty} {low:g}-{high:g}" for low, high in spec],
				return_dtype=pl.String,
			)
		spans.append(span.alias(name))

	return data.with_columns(spans)


def _collapse(mode: str, metric: str | None = None) -> pl.Expr:
	"""Value collapsed the way ``mode`` names, from ``metric``'s rows only when given."""
	value, exposure = pl.col(VALUE), pl.col(EXPOSURE)
	if metric is not None:
		value, exposure = (column.filter(pl.col(METRIC) == metric) for column in (value, exposure))

	rate = pl.when(exposure.sum() > 0).then(value.sum() / exposure.sum())
	return {"total": value.sum(), "exposure": exposure.sum(), "mean": value.mean()}.get(mode, rate)


def build_frame(
	frame: pl.LazyFrame,
	state: dict[str, Any],
	catalog: Sequence[Field],
) -> pl.DataFrame:
	"""Collapse the project table to one row per drawn point.

	Value is always grouped by Metric internally, even when Metric holds no shelf of
	its own, so a rate divides only after both halves are summed within one metric -
	the hourly rows then sum to a phase, a day or a whole recording without the
	mean-of-rates distortion. Metric only survives into the output when it is itself
	on a shelf; otherwise exactly one metric must be selected, which
	``warnings_for`` blocks on if it is not.

	A numeric ordered field named in ``state["bins"]`` is grouped in the blocks
	:func:`parse_bins` reads there rather than one group per value, so the field reads
	as ``day 1-3``, ``day 4-6`` - which is what a facet needs to compare one stretch of
	a recording to another. A block spec that cannot be read is ignored here and
	reported by :func:`warnings_for`. Grouped Age is first rounded to whole
	``state["age_unit"]``, after filtering, since its filter range is in days; blocks
	then count in that unit.

	Returns:
		The aggregated frame, with one column per field on a shelf.
	"""
	keys = group_keys(state, catalog)
	wanted = measures(state, catalog)
	singles = [item for item in wanted if item.metric is not None]
	filters = state.get("filters", {})
	if singles:
		# A single-metric chip names its own metric, so a Metric filter could only empty it.
		filters = {column: chosen for column, chosen in filters.items() if column != METRIC}
	frame = apply_filters(frame, filters)
	mode = state.get("measure_as", DEFAULT_MODE)
	hovered = hover_listed(state, catalog)
	carried: set[str] = set()

	per_unit = AGE_UNITS.get(state.get("age_unit", "days"), 1)
	if AGE in keys and per_unit != 1:
		frame = frame.with_columns(
			(pl.col(AGE) / per_unit).round(mode="half_away_from_zero").cast(pl.Int64)
		)

	if mode == "mean":
		# An hour's value is split over the positions it happened in; put it back
		# together first, or the mean would be over positions rather than hours.
		spread = {"position", "position_type"} - set(keys)
		# A position on Hover rides along as each hour's list, flattened when listed.
		carried = spread & set(hovered)
		frame = frame.group_by(cs.exclude(VALUE, EXPOSURE, *spread)).agg(
			pl.sum(VALUE), pl.sum(EXPOSURE), *carried
		)

	listed = [
		_listing(
			name,
			pl.col(name).list.explode(empty_as_null=False) if name in carried else pl.col(name),
		)
		for name in hovered
	]

	# Only what is grouped is binned: a leftover block for a field since taken off every
	# shelf would otherwise still rewrite its column.
	bins = {
		name: parsed
		for name, spec in state.get("bins", {}).items()
		if name in keys and (parsed := parse_bins(spec)) is not None
	}
	frame = apply_bins(frame, bins)

	def collected(grouped: pl.LazyFrame) -> pl.DataFrame:
		return label_bins(
			grouped.sort(keys).collect() if keys else grouped.collect(), bins, catalog
		)

	if not wanted:
		return collected(frame.group_by(keys).agg(pl.len().alias("rows"), *listed))

	value = next((item for item in wanted if item.agg == "metric" and item.metric is None), None)
	plain = [pl.mean(item.name) for item in wanted if item.agg != "metric"] + listed
	plain += [_collapse(mode, item.metric).alias(item.name) for item in singles]

	if value is None:
		return collected(frame.group_by(keys).agg(plain) if keys else frame.select(plain))

	agg_keys = keys if METRIC in keys else [*keys, METRIC]

	grouped = frame.group_by(agg_keys).agg(*plain, _collapse(mode).alias(value.name))

	if METRIC not in keys:
		grouped = grouped.drop(METRIC)

	return collected(grouped)


METRIC_UNITS: dict[str, str] = {
	"activity": "visits/hour",
	"time_alone": "fraction of time",
	"time_together": "fraction, per partner",
	"pairwise_encounters": "encounters/partner-hour",
	"n_chasing": "chasings/partner-hour",
	"n_chased": "chasings/partner-hour",
	"n_chasing_per_detection": "chasings/detection",
}

#: Shelves that give a faceted metric its own axis, so pooling several is not a problem.
FACETS: tuple[str, ...] = ("facet_row", "facet_col")


@dataclass(frozen=True)
class Note:
	"""One line for the banner above the plot.

	Attributes:
		text: the message.
		blocking: whether the plot cannot be drawn until this is resolved.
	"""

	text: str
	blocking: bool = False


def warnings_for(
	frame: pl.LazyFrame, state: dict[str, Any], catalog: Sequence[Field]
) -> list[Note]:
	"""Everything worth saying before the figure is trusted.

	Args:
		frame: read only for the metric count a pooling warning needs.

	Returns:
		Notes for the banner, blocking ones first.
	"""
	plot = plot_type(state["kind"])
	notes: list[Note] = []

	missing = [LABELS[channel] for channel in plot.required if not state["channels"].get(channel)]
	if missing:
		notes.append(Note(f"Drop a field on {', '.join(missing)} to draw the plot.", blocking=True))

	wanted = measures(state, catalog)
	pooled = any(item.agg == "metric" and item.metric is None for item in wanted)
	singles = [item.label for item in wanted if item.metric is not None]
	picked = state.get("filters", {}).get(METRIC) or metric_names(frame)
	faceted = any(METRIC in state["channels"].get(channel, []) for channel in FACETS)

	if singles and (pooled or METRIC in group_keys(state, catalog)):
		notes.append(
			Note(
				f"{', '.join(singles).capitalize()} each read one metric, so they cannot share the "
				"plot with Value or Metric, which range over all of them. Take Value and Metric "
				"off the shelves.",
				blocking=True,
			)
		)

	axes = {name for axis in ("x", "y") for name in state["channels"].get(axis, [])}
	if state["kind"] == "heatmap" and any(item.name in axes for item in wanted):
		notes.append(
			Note(
				"A heatmap draws one cell per X and Y value, so X and Y take categories or "
				"ordered fields and the measure goes on Z. Two measures against each other "
				"are a scatter.",
				blocking=True,
			)
		)

	if pooled and len(picked) > 1 and not faceted:
		units = "; ".join(
			f"{name} ({METRIC_UNITS[name]})" if name in METRIC_UNITS else name for name in picked
		)
		notes.append(
			Note(
				f"Value would pool {len(picked)} metrics with different units ({units}). "
				"Filter Metric to one, or put Metric on Facet row or Facet col so each gets "
				"its own axis.",
				blocking=True,
			)
		)

	if state["kind"] in DISTRIBUTIONS and not state["channels"].get(DETAIL):
		notes.append(
			Note(
				"Detail is empty, so every group collapses to one number and each box is "
				"drawn from a single point. Drop recording and animal id on Detail to spread "
				"it over animals."
			)
		)

	grouped = group_keys(state, catalog)
	unreadable = [
		name
		for name, spec in state.get("bins", {}).items()
		if name in grouped and parse_bins(spec) is None
	]
	if unreadable:
		notes.append(
			Note(
				f"Blocks for {', '.join(unreadable)} could not be read, so every value is "
				f"still its own group. Type a block size or spans: {BIN_HINT}."
			)
		)

	if state.get("measure_as") == "mean" and any(item.agg == "metric" for item in wanted):
		notes.append(
			Note(
				"An hourly mean weights every hour equally, however much of it was observed; "
				"the rate is the one that sums correctly over a phase or a day."
			)
		)

	return notes


def auto_facet_metric(state: dict[str, Any]) -> dict[str, Any]:
	"""Give Value its own axis per metric when several could still land on it.

	Called right after Value lands on a shelf. Dropping Value with more than one
	metric still selected, and Metric not already assigned anywhere, would otherwise
	trip the pooled-units warning; putting Metric on Facet row keeps each metric
	readable instead of waiting for that warning to say so. A plot type without a
	Facet row shelf is left alone.
	"""
	plot = plot_type(state["kind"])
	if "facet_row" not in plot.channels or state["channels"].get("facet_row"):
		return state

	if any(METRIC in names for names in state["channels"].values()):
		return state

	picked = state.get("filters", {}).get(METRIC)
	if isinstance(picked, list) and len(picked) == 1:
		return state

	state["channels"]["facet_row"] = [METRIC]
	return state


def placeholder(message: str) -> go.Figure:
	"""An empty canvas carrying ``message`` in the middle of it instead of a figure."""
	figure = go.Figure()
	figure.update_layout(
		template=TEMPLATE,
		xaxis={"visible": False},
		yaxis={"visible": False},
		margin={"l": 20, "r": 20, "t": 20, "b": 20},
		annotations=[
			{
				"text": message,
				"showarrow": False,
				"xref": "paper",
				"yref": "paper",
				"x": 0.5,
				"y": 0.5,
				"font": {"size": 15, "color": "#7f8ca5"},
			}
		],
	)

	return figure


def _excludes_outliers(state: dict[str, Any]) -> bool:
	return state["kind"] in POINTED and state.get("outliers") == "exclude"


def plotted_data(
	frame: pl.LazyFrame, state: dict[str, Any], catalog: Sequence[Field]
) -> pl.DataFrame | None:
	"""The rows :func:`build_figure` draws, as the plot's CSV export holds them.

	Columns are titled as the axes are. Excluded outliers stay in, flagged in an
	``excluded_outlier`` column, so the export shows what the plot's note counts.

	Returns:
		The rows, or ``None`` when the plot cannot be drawn or no row matches.
	"""
	if any(note.blocking for note in warnings_for(frame, state, catalog)):
		return None

	data = build_frame(frame, state, catalog)
	if data.is_empty():
		return None

	if _excludes_outliers(state):
		data = data.with_columns(outliers(state, catalog).alias("excluded_outlier"))
	mode = state.get("measure_as", DEFAULT_MODE)
	labels = {item.name: label_for(item, mode) for item in catalog}
	return data.rename({name: labels[name] for name in data.columns if name in labels})


def build_figure(
	frame: pl.LazyFrame,
	state: dict[str, Any],
	catalog: Sequence[Field],
) -> tuple[go.Figure, list[str]]:
	"""Aggregate and draw whatever the shelves currently describe.

	Returns:
		The figure and any warnings to show above it.
	"""
	plot = plot_type(state["kind"])
	notes = warnings_for(frame, state, catalog)
	blocking = next((note for note in notes if note.blocking), None)

	if blocking is not None:
		return placeholder(blocking.text), [note.text for note in notes if note is not blocking]

	data = build_frame(frame, state, catalog)
	if data.is_empty():
		return placeholder("No rows match the current filters."), [note.text for note in notes]

	if _excludes_outliers(state):
		total = data.height
		data = data.remove(outliers(state, catalog))
		if dropped := total - data.height:
			notes.append(
				Note(
					f"Excluded {dropped} of {total} points lying more than {FENCE:g} IQR outside "
					f"their {plot.label.lower()}'s quartiles."
				)
			)

	mode = state.get("measure_as", DEFAULT_MODE)
	labels = {item.name: label_for(item, mode) for item in catalog}
	kwargs: dict[str, Any] = {
		channel: list(names) if channel in MULTI else names[0]
		for channel, names in assigned(state).items()
		if channel not in (DETAIL, POINT_COLOR)
	}
	numeric = {item.name for item in measures(state, catalog)}
	if plot.name in OPTIONS:
		argument, choices = OPTIONS[plot.name]
		chosen = state.get(argument)
		kwargs[argument] = chosen if chosen in choices else next(iter(choices))
	if HOVER in kwargs:
		kwargs[HOVER] = {name: ":.4~r" if name in numeric else True for name in kwargs[HOVER]}

	fields = {item.name: item for item in catalog}
	point_field = next(iter(state["channels"].get(POINT_COLOR, [])), None)
	point_item = (
		fields.get(point_field) if plot.name in POINTED and point_field in data.columns else None
	)
	point_values: list = []
	if point_item is not None:
		# ROW leads the hover data, so it is customdata[0] on every trace plotly draws.
		data = data.with_row_index(ROW)
		hover = ":.4~r" if point_item.name in numeric else True
		kwargs[HOVER] = {ROW: False, **kwargs.get(HOVER, {}), point_item.name: hover}
		if point_item.kind != "measure":
			point_values = data[point_item.name].unique().sort(nulls_last=True).to_list()

	# Builder colours come from sample_palette(n) for the categories actually on the
	# Colour shelf, not the colorway's first n: plotly express walks the colorway in
	# order, so two categories would get neighbouring, hard-to-tell-apart samples. Point
	# categories take the samples after the boxes', so no box shares a colour with a point.
	color_field = kwargs.get("color")
	color_item = fields.get(color_field) if isinstance(color_field, str) else None
	boxes = 0
	if color_item is not None and color_item.kind == "dimension":
		boxes = data.select(pl.col(color_item.name).n_unique()).item()
	elif color_field is None and point_values:
		boxes = 1
	colorway = theme.sample_palette(boxes + len(point_values)) or None
	if boxes and colorway:
		kwargs["color_discrete_sequence"] = colorway[:boxes]

	# One heatmap cell per value: a whole-number axis stays numeric, so Format's bounds are
	# values, with bins one wide; anything else becomes categories, in sorted order taken
	# before the cast, or text would sort 10 before 2.
	unit_bins: dict[str, dict[str, float]] = {}
	if plot.name == "heatmap":
		texts = []
		for axis in ("x", "y"):
			column = data[kwargs[axis]].drop_nulls()
			if column.dtype.is_integer() and not column.is_empty():
				unit_bins[f"{axis}bins"] = {
					"start": column.min() - 0.5,
					"end": column.max() + 0.5,
					"size": 1,
				}
			else:
				texts.append(kwargs[axis])
		kwargs["category_orders"] = {
			name: data[name].drop_nulls().unique().sort().cast(pl.String).to_list()
			for name in texts
		}
		data = data.with_columns(pl.col(*texts).cast(pl.String)) if texts else data

	try:
		figure = plot.builder(data, **kwargs, **plot.extra, labels=labels, template=TEMPLATE)
	except Exception as error:  # plotly rejects plenty of otherwise sensible combinations
		return placeholder(f"{type(error).__name__}: {error}"), [note.text for note in notes]

	# The declared colorway is what tells Format this figure has categories to recolour.
	figure.update_layout(
		margin={"l": 60, "r": 20, "t": 40, "b": 40},
		legend={"title": None},
		colorway=colorway,
	)
	figure.for_each_annotation(lambda note: note.update(text=note.text.split("=")[-1]))
	if unit_bins:
		figure.update_traces(**unit_bins)
	# Plotly types an axis from its first trace's values, and a trace of all-null categories
	# (an event a group's recordings never had) reads as numeric, hiding every other trace.
	for axis, update in (("x", figure.update_xaxes), ("y", figure.update_yaxes)):
		dtype = data.schema.get(kwargs.get(axis)) if isinstance(kwargs.get(axis), str) else None
		if dtype is not None and not (dtype.is_numeric() or dtype.is_temporal()):
			update(type="category")

	if point_item is not None:
		colors = dict(zip(point_values, (colorway or [])[boxes:], strict=True)) or None
		color_points(figure, data, point_item.name, labels[point_item.name], colors)

	return figure, [note.text for note in notes]


def outliers(state: dict[str, Any], catalog: Sequence[Field]) -> pl.Expr:
	"""Whether a row's measure sits past the :data:`FENCE` of its own box or violin.

	Each box is one combination of the categories on the shelves that split boxes, so
	Detail, Hover and Point colour, which only say what a point is, are left out. The
	fences are computed once: recomputing them on what is left would find new outliers
	on every pass.
	"""
	fields = {item.name: item for item in catalog}
	shelved = {
		channel: [fields[name] for name in names if name in fields]
		for channel, names in assigned(state).items()
	}
	value = next(
		(item.name for axis in ("y", "x") for item in shelved.get(axis, []) if not item.discrete),
		None,
	)
	if value is None:
		return pl.lit(False)

	keys = [
		item.name
		for channel, items in shelved.items()
		if channel not in (DETAIL, HOVER, POINT_COLOR)
		for item in items
		if item.discrete
	]
	column = pl.col(value)
	# Linear quartiles, the method plotly draws a box's own quartiles with.
	q1, q3 = column.quantile(0.25, "linear"), column.quantile(0.75, "linear")
	fence = FENCE * (q3 - q1)
	outside = ~column.is_between(q1 - fence, q3 + fence)

	return outside.over(list(dict.fromkeys(keys))) if keys else outside


def color_points(
	figure: go.Figure,
	data: pl.DataFrame,
	name: str,
	label: str,
	colors: dict[Any, str] | None,
) -> None:
	"""Recolour every box's or violin's points by ``data[name]``, rows found through ``ROW``.

	A box or violin trace takes one marker colour, so each keeps its shape with its own
	points switched off and gets one point-only copy per colour drawn over it. A copy
	shares its box's offset group, so it lands in the same slot, and plotly jitters its
	points the way it would have the box's own.

	Args:
		colors: category to colour, listed on the legend; ``None`` cuts a measure into
			:data:`POINT_BANDS` steps of the house scale, read off a colorbar instead.
	"""
	styles: dict[Any, tuple[str, str, float | None]]
	if colors is None:
		column = pl.col(name)
		scaled = ((column - column.min()) / (column.max() - column.min())).fill_nan(0.0)
		keys = data.select((scaled * POINT_BANDS).floor().clip(0, POINT_BANDS - 1)).to_series()
		shades = [(step + 0.5) / POINT_BANDS for step in range(POINT_BANDS)]
		samples = px.colors.sample_colorscale(theme.AURORA, shades)
		styles = {
			step: ("", color, t)
			for step, (t, color) in enumerate(zip(shades, samples, strict=True))
		}
		styles[None] = ("no value", MISSING, None)
		low, high = data[name].min(), data[name].max()
	else:
		keys = data[name]
		styles = {key: (str(key), color, None) for key, color in colors.items()}
	keys = keys.to_list()

	boxed_legend = any(trace.showlegend is not False for trace in figure.data)
	listed: set = set()
	for index, trace in enumerate(list(figure.data)):
		if not isinstance(trace, go.Box | go.Violin):
			continue
		points = "boxpoints" if isinstance(trace, go.Box) else "points"
		# Unset offset groups each take their own slot, so give each a name to share.
		trace.offsetgroup = trace.offsetgroup or f"{POINT_COLOR}{index}"
		rows = [keys[row] for row in trace.customdata[:, 0].astype(int)]

		for key, (title, color, shade) in styles.items():
			pick = [i for i, value in enumerate(rows) if value == key]
			if not pick:
				continue
			copy = type(trace)(trace)
			copy.update(
				x=None if trace.x is None else [trace.x[i] for i in pick],
				y=None if trace.y is None else [trace.y[i] for i in pick],
				customdata=trace.customdata[pick],
				name=title,
				legendgroup=f"{POINT_COLOR}:{title}",
				showlegend=False,
				marker_color=color,
				fillcolor="rgba(0,0,0,0)",
				line_width=0,
				hoveron="points",
				# Lets Format resample the colour when it swaps the colour scale.
				meta={"dehScale": shade} if shade is not None else None,
				**{points: "all"},
			)
			if isinstance(copy, go.Violin):
				copy.update(meanline_visible=False, box_visible=False)
			figure.add_trace(copy)
			listed.add(key)
		if isinstance(trace, go.Box):
			# Hidden, not switched off: a box without points whiskers to its min and max
			# rather than to the 1.5 IQR fences.
			trace.update(marker_opacity=0, hoveron="boxes")
		else:
			trace.update(points=False)

	if colors is not None:
		# An invisible box draws an empty legend icon, so each colour is listed by an empty
		# dot trace instead; sharing the legend group, a click on it hides every box's copy.
		# Below the boxes' own entries in the one legend, two legends could only overlap.
		for first, key in enumerate(key for key in styles if key in listed):
			title, color, _ = styles[key]
			figure.add_trace(
				go.Scatter(
					x=[None],
					y=[None],
					mode="markers",
					marker_color=color,
					name=title,
					legendgroup=f"{POINT_COLOR}:{title}",
					legendgrouptitle_text=None if first else label,
					hoverinfo="skip",
				)
			)
		return

	# The colour bar shares the right margin with the boxes' legend, so it starts a quarter
	# down, leaving the legend room for about six entries above it.
	placed = {"len": 0.6, "y": 0.75, "yanchor": "top"} if boxed_legend else {}
	figure.update_layout(
		coloraxis={
			"colorscale": theme.AURORA,
			"cmin": low,
			"cmax": high,
			"colorbar": {"title": {"text": label}, **placed},
		}
	)
	# Box and violin markers cannot ride a colour axis, so this empty trace draws its bar.
	figure.add_trace(
		go.Scatter(
			x=[None],
			y=[None],
			mode="markers",
			marker={"color": [low, high], "coloraxis": "coloraxis"},
			showlegend=False,
			hoverinfo="skip",
		)
	)


def prune(state: dict[str, Any], keep: Iterable[str]) -> dict[str, Any]:
	"""Drop assignments the new plot type has no shelf for.

	Args:
		keep: the channels the new plot type supports.
	"""
	supported = set(keep)
	state["channels"] = {
		channel: names for channel, names in state["channels"].items() if channel in supported
	}

	return state
