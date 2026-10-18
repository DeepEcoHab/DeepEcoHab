import math
import textwrap
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import polars as pl
from _plotly_utils.utils import to_typed_array_spec as _array
from plotly.colors import sample_colorscale

from deepecohab.core.data_model import Layout
from deepecohab.plotting.animals import ColorMapping, collapse_legend
from deepecohab.plotting.export import LEGEND_ITEM_PX, text_px
from deepecohab.plotting.prepare import READ_KINDS, Heatmap
from deepecohab.plotting.registry import Figure
from deepecohab.plotting.theme import (
	AURORA,
	COLORBAR,
	CORNER_RADIUS,
	EVENT_LOOKS,
	EVENT_TINT,
	FONT_SIZE,
	LAGOON,
	PHASE_BAND,
	READ_LOOKS,
	sample_palette,
)

if TYPE_CHECKING:
	from polars._typing import PythonLiteral


def _tick_labels(names: list[str]) -> list[str]:
	"""Turn snake_case position names into axis tick text."""
	return [name.capitalize().replace("_", " ") for name in names]


def _shade(color: str) -> str:
	"""An ``rgb()`` colour at 20% opacity, for a band around its line."""
	return color.replace("rgb", "rgba").replace(")", ", 0.2)")


def _trace_rows(frame: pl.DataFrame, mapping: ColorMapping) -> dict[str, pl.DataFrame]:
	"""Each trace's rows, keyed by its :attr:`~ColorMapping.trace_column` value in trace order.

	A value with no rows has no entry.
	"""
	groups = frame.partition_by(mapping.trace_column, as_dict=True)

	return {value: groups[(value,)] for value in mapping.order if (value,) in groups}


def _axis(figure: Figure, name: str) -> dict:
	"""The layout's ``name`` axis, created empty if the figure has none yet."""
	return figure["layout"].setdefault(name, {})


def _add_shapes(figure: Figure, shapes: list[dict]) -> None:
	figure["layout"].setdefault("shapes", []).extend(shapes)


def _vline(x: float) -> dict:
	"""A thin line down the plot at ``x``, styled by the template's shape defaults."""
	return {
		"type": "line",
		"x0": x,
		"x1": x,
		"xref": "x",
		"y0": 0,
		"y1": 1,
		"yref": "y domain",
		"line": {"width": 1.5},
	}


def _line_traces(
	frame: pl.DataFrame,
	mapping: ColorMapping,
	x: str,
	y: str,
	shape: str = "spline",
	**extra: Any,
) -> list[dict]:
	"""One line per trace value, hovering its column values as plotly express labels them."""
	traces = []
	for value, rows in _trace_rows(frame, mapping).items():
		traces.append(
			{
				"type": "scatter",
				"name": value,
				"x": _array(rows[x]),
				"y": _array(rows[y]),
				"mode": "lines",
				"line": {"color": mapping.trace_colors[value], "shape": shape},
				"hovertemplate": (
					f"{mapping.trace_column}={value}<br>{x}=%{{x}}<br>{y}=%{{y}}<extra></extra>"
				),
			}
			| extra
		)

	return traces


def _band_segment(
	phase: str,
	x0: float,
	x1: float,
	*,
	xref: str = "x",
	y: tuple[float, float] = (1.01, 1.035),
) -> dict:
	"""One stretch of the phase band, in the light theme's colour for ``phase``."""
	colors = PHASE_BAND["light"]

	return {
		"type": "rect",
		"name": f"phase-band-{phase}",
		"xref": xref,
		"yref": "paper",
		"x0": x0,
		"x1": x1,
		"y0": y[0],
		"y1": y[1],
		"fillcolor": colors["dark_phase"] if phase == "dark_phase" else colors["light_phase"],
		"line": {"width": 0},
		"layer": "above",
	}


def _phase_band(
	figure: Figure,
	phases: dict[str, float],
	switch: float,
	span: tuple[float, float],
	*,
	xref: str = "x",
	y: tuple[float, float] = (1.01, 1.035),
) -> None:
	"""Draw the phases as a thin two-colour band along an hour axis.

	The same band the control bar's hours slider carries under its track, so a phase
	reads the same whether you are choosing hours or looking at a plot - which is what
	the "DARK" / "LIGHT" captions it replaces cost a caption's worth of space to say.

	A shape holds a literal colour, so these are drawn in the light theme's pair and
	repainted by :func:`~deepecohab.plotting.theme.apply` for the other one.

	Args:
		phases: phase onsets in hours since the first phase's onset.
		switch: where the second phase starts, in the x axis's own units. Clamped into
			``span``, so a narrowed hours window leaves whichever phase it kept.
		span: first and last x, so the band runs the full width of the panel.
		xref: which x axis to place it on, for a figure with more than one.
		y: the band's edges in paper coordinates. Above the panel by default; a faceted
			figure passes a pair below it, where its shared axis is.
	"""
	order = sorted(phases, key=lambda name: phases[name])
	edges = [span[0], min(max(switch, span[0]), span[1]), span[1]] if len(order) > 1 else list(span)

	_add_shapes(
		figure,
		[
			_band_segment(name, x0, x1, xref=xref, y=y)
			for name, x0, x1 in zip(order, edges[:-1], edges[1:], strict=True)
		],
	)


def _phase_markers(figure: Figure, phases: dict[str, float], hours: pl.Series) -> None:
	"""Mark the phase switch on an hour axis that begins at the first phase's onset.

	A single thin line, styled by the active template's shape defaults so it reads
	on either ground without competing with the Phase-coloured traces, and the phase
	band above the panel naming which side is which.
	"""
	switch = max(phases.values())

	_add_shapes(figure, [_vline(switch)])
	# From the data rather than the full day: an hours window narrows these axes.
	if not hours.is_empty():
		_phase_band(figure, phases, switch, (hours.min() - 0.5, hours.max() + 0.5))  # ty: ignore[unsupported-operator]

	_axis(figure, "xaxis")["dtick"] = 1


def _pin_bin_axis(figure: Figure, bins: pl.Series) -> None:
	"""Hold a binned x axis to its bins, half a bin either side.

	Autorange counts what :func:`_event_spans` draws: a span reaches half a bin past the
	outermost bin, so the axis would run past the data on its own. Pinning the range
	first is also what the bin columns want - a band drawn on the edge bin would
	otherwise push the axis out the same way.
	"""
	_axis(figure, "xaxis")["range"] = [(bins - 0.5).min(), (bins + 0.5).max()]


#: Pixel pitch of one event lane above a plot, and the height of the bar inside it.
_LANE_PX = 14
_LANE_BAR_PX = 6

#: Event names sit in the left gutter as row labels, cut to fit; the hover has the full name.
_LABEL_CHARS = 14
_LABEL_FONT_PX = 11
_LABEL_GAP_PX = 6

#: Room a faceted figure's top panel title takes above the panel, in pixels.
_FACET_TITLE_PX = 20


def _event_runs(spans: pl.DataFrame) -> pl.DataFrame:
	"""Fuse each event's overlapping or touching spans, joining the cages they cover.

	``spans`` carries ``index``, the event's place among the recording's events.
	"""
	return (
		spans.sort("x0")
		.with_columns(
			run=(pl.col("x0") > pl.col("x1").cum_max().shift())
			.fill_null(True)
			.cum_sum()
			.over("index")
		)
		.group_by("event", "index", "run", maintain_order=True)
		.agg(
			pl.col("x0").min(),
			pl.col("x1").max(),
			*[
				getattr(pl.col(c), f)()
				for c, f in (("start", "min"), ("end", "max"))
				if c in spans.columns
			],
			pl.col("position").drop_nulls().unique().sort().str.join(", "),
		)
		.drop("run")
	)


def _event_union(spans: pl.DataFrame) -> pl.DataFrame:
	"""Fuse overlapping or touching spans whatever their event: one mark per busy stretch."""
	return _event_runs(spans.with_columns(event=pl.lit(""), index=pl.lit(0)))


def _event_lanes(figure: Figure, runs: pl.DataFrame, xrefs: list[str], offset: float) -> None:
	"""Draw one lane per event above the plot, a bar wherever the event ran.

	The same picture as the events card's strip: lanes stack upwards in the recording's
	event order from ``offset`` pixels above the phase band (or the plot top without one),
	each named in the left gutter, clear of its bars, the name's hover listing its cages.
	Pixel-sized, so a lane reads the same on a short card as on a tall one.
	"""
	layout = figure["layout"]
	anchor = max(
		(
			shape["y1"]
			for shape in layout.get("shapes", [])
			if shape.get("name", "").startswith("phase-band") and shape["y1"] > 1
		),
		default=1,
	)

	lanes = runs.sort("index", "x0").partition_by("index", maintain_order=True)
	bars, labels, tips = [], [], []
	for lane, rows in enumerate(lanes):
		index = rows["index"][0]
		color = EVENT_LOOKS[index % len(EVENT_LOOKS)]
		y0 = offset + lane * _LANE_PX + (_LANE_PX - _LANE_BAR_PX) / 2
		bars += [
			{
				"type": "rect",
				"name": "event-lane",
				"xref": xref,
				"x0": row["x0"],
				"x1": row["x1"],
				"yref": "paper",
				"ysizemode": "pixel",
				"yanchor": anchor,
				"y0": y0,
				"y1": y0 + _LANE_BAR_PX,
				"fillcolor": color,
				# A 10-minute bout in a week-long recording is a sliver; the stroke keeps
				# it visible, as the events card's minimum bar width does.
				"line": {"color": color, "width": 3},
				"layer": "above",
			}
			for row in rows.iter_rows(named=True)
			for xref in xrefs
		]

		name = str(rows["event"][0])
		# One tip per bar, in the order of the bars above
		# the position list is left out for the whole habitat.
		tips += [
			"\n".join(
				[
					name,
					*(
						[f"{row['start']:%Y-%m-%d %H:%M} - {row['end']:%Y-%m-%d %H:%M}"]
						if "start" in row
						else []
					),
					*(
						[", ".join(_tick_labels(row["position"].split(", ")))]
						if row["position"]
						else []
					),
				]
			)
			for row in rows.iter_rows(named=True)
			for _ in xrefs
		]
		cages = sorted({*joined.split(", ") for joined in rows["position"] if joined})
		labels.append(
			{
				"name": "event-label",
				"text": name if len(name) <= _LABEL_CHARS else f"{name[: _LABEL_CHARS - 1]}…",
				"hovertext": "<br>".join([name, *_tick_labels(cages)]),
				"x": 0,
				"xref": "paper",
				"xanchor": "right",
				"xshift": -_LABEL_GAP_PX,
				"y": anchor,
				"yref": "paper",
				"yshift": y0 + _LANE_BAR_PX / 2,
				"yanchor": "middle",
				"showarrow": False,
				"font": {"size": _LABEL_FONT_PX},
			}
		)

	layout.setdefault("annotations", []).extend(labels)
	_add_shapes(figure, bars)
	# Plotly's default top margin was sized for the title alone; a card sets its own.
	margin = layout.setdefault("margin", {})
	# The margin without lanes, for the dashboard to fall back to when events are hidden.
	meta = layout.setdefault("meta", {})
	meta.setdefault("dehBareMargin", dict(margin))
	meta["dehEventTips"] = tips
	margin["t"] = (margin.get("t") or 100) + offset + len(lanes) * _LANE_PX
	# Annotations don't drive automargin, so the left margin is widened for the names by hand.
	widest = max(text_px(label["text"], _LABEL_FONT_PX) for label in labels)
	margin["l"] = max(margin.get("l") or 80, widest + 2 * _LABEL_GAP_PX)


def _event_spans(
	figure: Figure,
	spans: pl.DataFrame,
	facets: list[str] | None = None,
	*,
	above: bool = False,
	offset: PythonLiteral | None = None,
) -> None:
	"""Mark where each event ran on a time axis: a lane above the plot, a mark inside it.

	Colour lives in the lanes alone - an event's comes from its place among the recording's
	events, so it matches on every plot and on the events card. Inside the plot every event
	is drawn alike, so overlapping events darken rather than mix hues, and none clashes with
	the data: a neutral wash behind the traces, or where ``above`` asks - a plot whose
	opaque traces would hide a wash behind them and turn grey under one in front - dotted
	lines at its ends. On a faceted figure - one heatmap panel per cage,
	where a wash would tint the colour scale - a span is outlined instead, on its cage's
	panel only when it has one; :func:`~deepecohab.plotting.theme.apply` colours the lines.

	``offset`` is taken off every x: a category axis places a bin at its index among the
	categories, not at its value.
	"""
	if spans.is_empty():
		return

	if offset is not None:
		spans = spans.with_columns(pl.col("x0", "x1") - offset)
	# An Enum's physical code is the event's place among the recording's events.
	spans = spans.with_columns(index=pl.col("event").to_physical())
	runs = _event_runs(spans)

	if facets is None:
		box = {"name": "event-span", "xref": "x", "yref": "y domain", "y0": 0, "y1": 1}
		if above:
			# Lines at every event's ends would bunch up where events overlap, so they mark
			# where a stretch of overlapping events starts and stops; the lanes tell them apart.
			marks = [
				{"type": "line", "x0": x, "x1": x, "line": {"width": 1.5, "dash": "dot"}}
				for run in _event_union(spans).iter_rows(named=True)
				for x in (run["x0"], run["x1"])
			]
			layer = "above"
		else:
			marks = [
				{
					"type": "rect",
					"x0": run["x0"],
					"x1": run["x1"],
					"fillcolor": EVENT_TINT,
					"line": {"width": 0},
				}
				for run in runs.iter_rows(named=True)
			]
			layer = "below"
		_add_shapes(figure, [mark | box | {"layer": layer} for mark in marks])
		_event_lanes(figure, runs, ["x"], 2)
		return

	panels = list(zip(facets, figure["data"], strict=True))
	_add_shapes(
		figure,
		[
			{
				"type": "rect",
				"name": "event-span",
				"xref": trace["xaxis"],
				"yref": f"{trace['yaxis']} domain",
				"x0": span["x0"],
				"x1": span["x1"],
				"y0": 0,
				"y1": 1,
				"line": {"width": 1.5},
				"layer": "above",
			}
			for facet, trace in panels
			# Overlapping events in one cage share an outline, as they share a stretch of lines.
			for span in _event_union(
				spans.filter(pl.col("position").is_null() | (pl.col("position") == facet))
			).iter_rows(named=True)
		],
	)

	# Lanes run over the top row of panels, above its titles.
	tops = {
		trace["xaxis"]: figure["layout"][f"yaxis{trace['yaxis'][1:]}"]["domain"][1]
		for _, trace in panels
	}
	_event_lanes(
		figure,
		runs,
		[xaxis for xaxis, top in tops.items() if top == max(tops.values())],
		_FACET_TITLE_PX,
	)


#: Most panels a faceted heatmap puts in one column (stacked) or one row (grid).
FACETS_PER_LINE = 4


def _crosshair(figure: Figure) -> Figure:
	"""Trace the hovered cell's row and column across the plot."""
	spikes = {"showspikes": True, "spikemode": "across", "spikethickness": 1, "spikedash": "solid"}
	for name, axis in figure["layout"].items():
		if name.startswith(("xaxis", "yaxis")):
			axis.update(spikes)

	return figure


#: Trace meta that has rounded_corners.js draw the cells as rounded tiles with proportional gaps.
TILES = "tiles"


def _faceted_heatmap(
	heatmap: Heatmap,
	title: str,
	x_title: str,
	y_title: str,
	hover: tuple[str, str],
	*,
	square: bool = False,
	grid: bool = False,
) -> Figure:
	"""Draw one heatmap panel per facet, in reading order.

	Stacked one per row by default, sharing the x axis - the panels are then a day or
	an hour against animals, so lining their columns up matters. ``grid`` instead lays the
	panels out in a row, for panels - like a pairwise matrix - with no axis to share across
	facets; a row of four cages fits a full-width card without the dead space a square
	panel leaves when its card is taller than it is wide. Either way a line holds at most
	:data:`FACETS_PER_LINE` panels and a larger habitat wraps into more columns (stacked)
	or rows (grid). Every facet has the same labels, so only the outer panels carry tick
	labels.

	A single shared colour axis keeps the panels comparable, which is also why these
	figures offer one scope - cages or tunnels - at a time (§ Cage / tunnel scope).

	Panels are laid out on the grid ``plotly.subplots.make_subplots`` would draw: every
	cell gets its axes, numbered in reading order, and a stacked panel's x axis follows
	the bottom one of its column.
	"""
	n = len(heatmap.facets)
	cols = min(n, FACETS_PER_LINE) if grid else math.ceil(n / FACETS_PER_LINE)
	rows = math.ceil(n / cols)
	h_gap, v_gap = 0.06, min(0.12, 1 / max(rows - 1, 1))
	width, height = (1 - h_gap * (cols - 1)) / cols, (1 - v_gap * (rows - 1)) / rows

	def suffix(cell: int) -> str:
		return "" if cell == 0 else str(cell + 1)

	layout: dict[str, Any] = {}
	names, titles = _tick_labels(heatmap.facets), []
	for cell in range(rows * cols):
		row, col = divmod(cell, cols)
		left, bottom = col * (width + h_gap), (rows - 1 - row) * (height + v_gap)
		x_axis = {"anchor": f"y{suffix(cell)}", "domain": [left, left + width], "automargin": True}
		y_axis: dict[str, Any] = {
			"anchor": f"x{suffix(cell)}",
			"domain": [bottom, bottom + height],
			"automargin": True,
		}
		if not grid and row < rows - 1:
			x_axis |= {"matches": f"x{suffix((rows - 1) * cols + col)}", "showticklabels": False}
		if row == rows - 1 and col == 0:
			x_axis["title"] = {"text": x_title}
		if col == 0:
			y_axis["title"] = {"text": y_title}
		if square:
			# Each panel is animal-by-animal; a shared scale keeps its cells square instead
			# of stretched across the panel's width. Shrink a panel to its cells, under its
			# title, rather than padding empty rows around it, and toward the colour bar
			# rather than leaving a gap before it.
			x_axis |= {"constrain": "domain", "constraintoward": "right"}
			y_axis |= {"constrain": "domain", "constraintoward": "top"}
			if cell < n:
				y_axis |= {"scaleanchor": f"x{suffix(cell)}", "scaleratio": 1}
		if cell < n:
			titles.append(
				{
					"text": f"<b>{names[cell]}</b>",
					"x": left + width / 2,
					"xref": "paper",
					"xanchor": "center",
					"y": bottom + height,
					"yref": "paper",
					"yanchor": "bottom",
					"showarrow": False,
					"font": {"size": 16},
				}
			)
		layout[f"xaxis{suffix(cell)}"], layout[f"yaxis{suffix(cell)}"] = x_axis, y_axis

	value = "%{z}" if heatmap.text is None else "%{customdata}"
	hovertemplate = f"{hover[0]}: %{{x}}<br>{hover[1]}: %{{y}}<br>Value: {value}<extra></extra>"
	data = []
	for index, values in enumerate(heatmap.values):
		col = index % cols
		trace = {
			"type": "heatmap",
			"z": _array(values),
			"x": [str(label) for label in heatmap.x],
			"y": heatmap.y,
			"coloraxis": "coloraxis",
			"hovertemplate": hovertemplate,
			"xaxis": f"x{suffix(index)}",
			"yaxis": f"y{suffix(index)}",
			"meta": TILES,
		}
		if heatmap.text is not None:
			trace["customdata"] = heatmap.text[index]
		data.append(trace)

		show_x = index + cols >= n
		show_y = col == 0
		layout[f"xaxis{suffix(index)}"] |= {
			"showticklabels": show_x,
			"ticks": "outside" if show_x else "",
		}
		layout[f"yaxis{suffix(index)}"] |= {
			"showticklabels": show_y,
			"ticks": "outside" if show_y else "",
		}

	layout |= {
		"annotations": titles,
		"title": {"text": title},
		"coloraxis": {
			"cmin": 0,
			"colorscale": LAGOON,
			"colorbar": {"title": {"text": heatmap.label}},
		},
	}

	return _crosshair(Figure(data=data, layout=layout))


def _position_plot(
	frame: pl.DataFrame,
	positions: list[str],
	mapping: ColorMapping,
	*,
	y: str,
	title: str,
	x_title: str,
	y_title: str,
	hover: list[str],
	agg: Literal["sum", "mean"],
	granularity: str,
) -> Figure:
	"""Per-animal values against position: grouped bars for a sum, boxes for a mean.

	``hover`` names the columns the hover lists after the position, ``y`` among them
	when its value is worth showing as it is.
	"""
	extra = [column for column in hover if column != y]
	if agg == "mean":
		extra.append(granularity)
	labels = [f"{mapping.trace_column}=%{{customdata[0]}}", "position=%{x}"]
	labels += [f"{y}=%{{y}}"] if y in hover else []
	labels += [f"{column}=%{{customdata[{i}]}}" for i, column in enumerate(extra, 1)]
	hovertemplate = "<br>".join(labels) + "<extra></extra>"

	data = []
	for value, rows in _trace_rows(frame, mapping).items():
		trace = {
			"name": value,
			"x": _array(rows["position"]),
			"y": _array(rows[y]),
			"customdata": rows.select(mapping.trace_column, *extra).rows(),
			"hovertemplate": hovertemplate,
		}
		color = mapping.trace_colors[value]
		if agg == "sum":
			trace |= {"type": "bar", "marker": {"color": color, "line": {"width": 0}}}
		else:
			trace |= {
				"type": "box",
				"marker": {"color": color},
				"boxpoints": "outliers",
				"boxmean": True,
			}
		data.append(trace)

	figure = Figure(
		data=data,
		layout={
			"title": {"text": title},
			"xaxis": {
				"title": {"text": x_title},
				# Pinned so the index-based tick labels name the right category.
				"categoryorder": "array",
				"categoryarray": positions,
				"tickvals": list(range(len(positions))),
				"ticktext": _tick_labels(positions),
			},
			"yaxis": {"title": {"text": y_title}},
			"legend": {"tracegroupgap": 0},
			"barmode" if agg == "sum" else "boxmode": "group",
			"barcornerradius": CORNER_RADIUS,
		},
	)
	collapse_legend(figure, mapping)

	return figure


def plot_activity(
	frame: pl.DataFrame,
	positions: list[str],
	mapping: ColorMapping,
	metric: Literal["visits", "time"],
	agg: Literal["sum", "mean"],
	granularity: str,
	value_label: str,
	place: str = "position",
) -> Figure:
	"""Plots bar or box graph of cage and tunnel visits or time spent."""
	return _position_plot(
		frame,
		positions,
		mapping,
		y=metric,
		title="<b>Visits to each position</b>"
		if metric == "visits"
		else "<b>Time spent in each position</b>",
		x_title=f"<b>{place.capitalize()}</b>",
		y_title="<b>Number of visits</b>" if metric == "visits" else value_label,
		hover=["visits"] if metric == "visits" else ["time_text"],
		agg=agg,
		granularity=granularity,
	)


def plot_time_alone(
	frame: pl.DataFrame,
	positions: list[str],
	mapping: ColorMapping,
	agg: Literal["mean", "sum"],
	granularity: str,
	value_label: str,
	place: str = "cage",
) -> Figure:
	"""Plot time alone per position as a bar or box plot."""
	return _position_plot(
		frame,
		positions,
		mapping,
		y="time_alone",
		title="<b>Time spent alone</b>",
		x_title=f"<b>{place.capitalize()}</b>",
		y_title=value_label,
		hover=["time_alone_text"],
		agg=agg,
		granularity=granularity,
	)


#: Title, y-axis title and hover label of the count line plots, per input.
_LINE_LABELS: frozendict[str, tuple[str, str, str]] = frozendict(
	{
		"activity": ("<b>Activity over time</b>", "<b>Antenna detections</b>", "Detections"),
		"chasings": ("<b>Chasing over time</b>", "<b># of chasing events</b>", "Events"),
	}
)

#: Hover label and axis title of each binned x axis.
_X_LABELS: frozendict[str, tuple[str, str]] = frozendict(
	{
		"hour": ("Hour", "<b>Hour since phase onset</b>"),
		"day": ("Day", "<b>Day</b>"),
		"phase_count": ("Phase", "<b>Phase</b>"),
	}
)


def _line_x_axis(
	figure: Figure, bins: pl.Series, phases: dict[str, float], spans: pl.DataFrame
) -> None:
	"""Title and pin a count line's x axis, marking the phase switch on the hour axis only."""
	axis = _axis(figure, "xaxis")
	axis["title"] = {"text": _X_LABELS[bins.name][1]}
	if bins.name == "hour":
		_pin_bin_axis(figure, bins)
		_phase_markers(figure, phases, bins)
		_event_spans(figure, spans)
	else:
		# A category axis ticks whole steps only, widening them as the window grows. It sits
		# a bin at its index among the categories, not its value, and so does a span.
		axis |= {"type": "category", "range": [0, bins.n_unique() - 1]}
		_event_spans(figure, spans, offset=bins.min())


#: Thickness of a side-panel bar or box, in category units.
_SIDE_WIDTH = 0.64


def _side_panel(
	figure: Figure,
	frame: pl.DataFrame,
	mapping: ColorMapping,
	agg: Literal["sum", "mean"],
	label: str,
) -> None:
	"""Sum up each trace in a narrow panel right of the lines: its total, or its means' spread.

	One row per trace, grouped top to bottom in legend order. Autorange happens in the
	browser, so the line axis gets its range pinned here - the band included, 5% of headroom
	either side - and the category range is solved from it so the lowest row's bottom edge
	sits on the line's zero.
	"""
	value = "total" if agg == "sum" else "mean"
	drawn = ["total"] if agg == "sum" else ["mean", "lower", "upper"]

	low, high = frame.select(
		pl.min_horizontal(drawn).min().alias("low"), pl.max_horizontal(drawn).max().alias("high")
	).row(0)
	low, high = min(low or 0, 0), max(high or 0, 0)
	pad = (high - low) * 0.05 or 1
	zero_at = (pad - low) / (high - low + 2 * pad)

	def category(row: str) -> str:
		return mapping.category_by_animal.get(row, row)

	legend = list(dict.fromkeys(category(row) for row in mapping.order))
	groups = _trace_rows(frame, mapping)
	rows = sorted(groups, key=lambda row: legend.index(category(row)))

	for row in rows:
		values = groups[row][value]
		side = {
			"xaxis": "x2",
			"yaxis": "y2",
			"name": row,
			"legendgroup": category(row),
			"showlegend": False,
			"orientation": "h",
			"width": _SIDE_WIDTH,
		}
		if agg == "sum":
			trace = {
				"type": "bar",
				"y": [row],
				"x": [values.sum()],
				"marker": {"color": mapping.trace_colors[row]},
				"hovertemplate": f"{row}: %{{x:,}}<extra></extra>",
			}
		else:
			trace = {
				"type": "box",
				"y": [row] * len(values),
				"x": values.to_list(),
				"boxmean": True,
				"line": {"color": mapping.trace_colors[row]},
				"boxpoints": "all",
				"jitter": 0.4,
				"pointpos": 0,
				"marker": {"size": 3},
			}
		figure["data"].append(side | trace)

	lowest_edge, top_edge = len(rows) - 1 + _SIDE_WIDTH / 2, -0.5
	_axis(figure, "xaxis")["domain"] = [0, 0.9]
	_axis(figure, "yaxis")["range"] = [low - pad, high + pad]
	figure["layout"] |= {
		"xaxis2": {
			"domain": [0.91, 1],
			"anchor": "y2",
			"title": {"text": f"<b>{'Total' if agg == 'sum' else 'Mean'} {label.lower()}</b>"},
			"labelalias": {"0": ""},
		},
		"yaxis2": {
			"anchor": "x2",
			"type": "category",
			"showticklabels": False,
			"range": [(lowest_edge - zero_at * top_edge) / (1 - zero_at), top_edge],
		},
		"barcornerradius": CORNER_RADIUS,
	}


def plot_count_line(
	frame: pl.DataFrame,
	mapping: ColorMapping,
	input_type: Literal["activity", "chasings"],
	x: Literal["hour", "day", "phase_count"],
	agg: Literal["sum", "mean"],
	phases: dict[str, float],
	spans: pl.DataFrame,
) -> Figure:
	"""Plots activity or chasings per hour of the day or per window unit.

	A sum draws each trace's totals; a mean draws its means in an SEM band.
	"""
	title, y_axes_label, hover_label = _LINE_LABELS[input_type]

	match agg:
		case "sum":
			data = _line_traces(frame, mapping, x, "total")
		case "mean":
			data = []
			x_label = _X_LABELS[x][0]
			for trace_value, rows in _trace_rows(frame, mapping).items():
				color = mapping.trace_colors[trace_value]
				category = mapping.category_by_animal.get(trace_value, trace_value)
				bins = rows[x].to_list()
				data += [
					{
						"type": "scatter",
						"x": bins + bins[::-1],
						"y": rows["upper"].to_list() + rows["lower"].to_list()[::-1],
						"fill": "toself",
						"fillcolor": _shade(color),
						"line": {"shape": "spline", "color": "rgba(255,255,255,0)"},
						"mode": "lines",
						"showlegend": False,
						"name": trace_value,
						"hoverinfo": "skip",
					},
					{
						"type": "scatter",
						"x": bins,
						"y": rows["mean"].to_list(),
						"mode": "lines",
						"line": {"shape": "spline", "color": color},
						"name": trace_value,
						"hovertemplate": (
							f"{mapping.legend_title}: {category}<br>{x_label}: %{{x}}<br>"
							f"{hover_label}: %{{y:.1f}}<extra></extra>"
						),
					},
				]

	figure = Figure(
		data=data,
		layout={
			"title": {"text": title},
			"yaxis": {"title": {"text": y_axes_label}},
			"legend": {"tracegroupgap": 0},
		},
	)
	collapse_legend(figure, mapping)
	_line_x_axis(figure, frame[x], phases, spans)
	_side_panel(figure, frame, mapping, agg, hover_label)

	return figure


def plot_ranking_line(frame: pl.DataFrame, mapping: ColorMapping, spans: pl.DataFrame) -> Figure:
	"""Plots line graph of ranking over time."""
	data = _line_traces(frame, mapping, "datetime", "ordinal", shape="hv")
	# As plotly express draws it: WebGL past 1000 points, where SVG starts to lag.
	if frame.height > 1000:
		for trace in data:
			trace["type"] = "scattergl"

	figure = Figure(
		data=data,
		layout={
			"title": {"text": "<b>Social dominance ranking in time</b>"},
			"xaxis": {"title": {"text": "<b>Timeline</b>"}, "type": "date"},
			"yaxis": {"title": {"text": "<b>Ranking</b>"}},
			"legend": {"tracegroupgap": 0},
			"margin": {"t": 60},
		},
	)
	collapse_legend(figure, mapping)
	_event_spans(figure, spans)

	return figure


def plot_ranking_distribution(frame: pl.DataFrame, mapping: ColorMapping) -> Figure:
	"""Plots line graph of ranking distribution with shaded area."""
	data = _line_traces(frame, mapping, "ranking", "probability_density", fill="tozeroy")

	figure = Figure(
		data=data,
		layout={
			"title": {"text": "<b>Ranking probability distribution</b>"},
			"xaxis": {"title": {"text": "<b>Ranking</b>"}},
			"yaxis": {"title": {"text": "<b>Probability density</b>"}},
			"legend": {"tracegroupgap": 0},
			"margin": {"t": 60},
		},
	)
	collapse_legend(figure, mapping)

	return figure


def plot_ranking_stability(
	frame: pl.DataFrame,
	mapping: ColorMapping,
	granularity: str,
	spans: pl.DataFrame,
) -> Figure:
	"""Plots animal rank on a per day (or per phase) basis."""
	cadence = "Daily" if granularity == "day" else "Per-phase"
	data = _line_traces(frame.sort(granularity), mapping, granularity, "rank", mode="lines+markers")

	figure = Figure(
		data=data,
		layout={
			"title": {"text": f"<b>{cadence} dominance rank trajectories</b>", "x": 0.5},
			"xaxis": {"title": {"text": _X_LABELS[granularity][1]}, "dtick": 1},
			"yaxis": {
				"title": {"text": "<b>Rank</b>"},
				"autorange": "reversed",
				"type": "category",
				"categoryorder": "array",
				"categoryarray": _array(frame["rank"].unique().sort()),
			},
			"legend": {"tracegroupgap": 1},
		},
	)
	collapse_legend(figure, mapping)
	_event_spans(figure, spans)

	return figure


def plot_time_spent_per_cage(
	heatmap: Heatmap,
	x: Literal["hour", "day", "phase_count"],
	spans: pl.DataFrame,
	place: str = "cage",
) -> Figure:
	"""Plots one heatmap per position of time spent, by hour or by window unit."""
	hover_label, x_title = _X_LABELS[x]
	title = (
		f"<b>Time spent per {place}</b>"
		if x == "hour"
		else f"<b>{place.capitalize()} preference over time</b>"
	)

	figure = _faceted_heatmap(
		heatmap, title, x_title, "<b>Animal ID</b>", (hover_label, "Animal ID")
	)
	# The x axis is categorical, so a bin sits at its index among the columns, not its value.
	_event_spans(figure, spans, heatmap.facets, offset=heatmap.x[0])

	return figure


def _matrix(
	matrix: np.ndarray, x: list[str], y: list[str], title: str, hovertemplate: str
) -> Figure:
	"""One heatmap of ``matrix``, its rows top to bottom and cells square, from zero up."""
	return Figure(
		data=[
			{
				"type": "heatmap",
				"z": _array(matrix),
				"x": x,
				"y": y,
				"coloraxis": "coloraxis",
				"hovertemplate": hovertemplate,
				"meta": TILES,
			}
		],
		layout={
			"title": {"text": title},
			"xaxis": {"scaleanchor": "y", "constrain": "domain"},
			"yaxis": {"autorange": "reversed", "constrain": "domain"},
			"coloraxis": {
				"colorscale": LAGOON,
				"cmin": 0,
				"cmax": np.nanmax(matrix).item(),
				"autocolorscale": False,
			},
		},
	)


def plot_heatmap(
	matrix: np.ndarray,
	animals: list[str],
	title: str,
	hover: tuple[str, str, str],
) -> Figure:
	"""Plots a square animal-by-animal matrix.

	Args:
		matrix: rows and columns in ``animals`` order.
		animals: axis labels for both rows and columns.
		title: figure title.
		hover: hover labels for the column, the row and the value.
	"""
	column, row, value = hover
	figure = _matrix(
		matrix,
		animals,
		animals,
		title,
		f"{column}: %{{x}}<br>{row}: %{{y}}<br>{value}: %{{z}}<extra></extra>",
	)
	# Square cells leave spare width; it goes left of the matrix, not between it and the colour bar.
	_axis(figure, "yaxis")["automargin"] = True
	_axis(figure, "xaxis").update(automargin=True, constraintoward="right")

	return _crosshair(figure)


def plot_sociability_heatmap(
	heatmap: Heatmap,
	metric: Literal["pairwise_encounters", "time_together"],
) -> Figure:
	"""Plots one pair matrix per cage, of meetings or time spent together."""
	title = (
		"<b>Number of pairwise encounters</b>"
		if metric == "pairwise_encounters"
		else "<b>Time spent together</b>"
	)

	return _faceted_heatmap(heatmap, title, "", "", ("X", "Y"), square=True, grid=True)


def plot_metrics_polar(frame: pl.DataFrame, mapping: ColorMapping) -> Figure:
	"""Plots mean z-scores of metrics with SEM shading as a polar plot."""
	data = []

	for trace_value, group in _trace_rows(frame, mapping).items():
		color = mapping.trace_colors[trace_value]
		group_closed = pl.concat([group, group.head(1)])
		polar = {"type": "scatterpolar", "theta": _array(group_closed["metric"]), "mode": "lines"}
		band = {
			"line": {"width": 0, "color": color, "shape": "spline"},
			"showlegend": False,
			"hoverinfo": "skip",
			"name": trace_value,
		}

		data += [
			polar | band | {"r": _array(group_closed["lower"])},
			polar
			| band
			| {"r": _array(group_closed["upper"]), "fill": "tonext", "fillcolor": _shade(color)},
			polar
			| {
				"r": _array(group_closed["mean"]),
				"line": {"color": color, "shape": "spline"},
				"name": trace_value,
			},
		]

	means = frame["mean"]
	metrics = frame["metric"].unique(maintain_order=True).to_list()
	aliases = {metric: "<br>".join(textwrap.wrap(metric, 14)) for metric in metrics}
	# Polar axes have no automargin: the angular labels hang past the circle, so the
	# margins make room for them and the legend is pinned beyond the right-hand ones.
	label_px = max((text_px(alias, FONT_SIZE) for alias in aliases.values()), default=0) + FONT_SIZE
	legend_px = (
		max(text_px(name, FONT_SIZE) for name in [mapping.legend_title, *mapping.categories])
		+ LEGEND_ITEM_PX
	)
	figure = Figure(
		data=data,
		layout={
			"title": {"text": "<b>Animal feature overview</b>", "y": 0.95, "x": 0.45},
			"polar": {
				"bgcolor": "rgba(0,0,0,0)",
				"angularaxis": {"labelalias": aliases},
				"radialaxis": {
					"visible": True,
					# Between the first two metrics, clear of the label at 0°.
					"angle": 180 / max(len(metrics), 1),
					# Series.min/max is typed as a broad union; arithmetic is valid for this column.
					"range": [means.min() - 0.5, means.max() + 0.5],  # ty: ignore[unsupported-operator]
				},
			},
			"margin": {"l": label_px, "r": label_px + legend_px},
			"legend": {"tracegroupgap": 0, "xref": "container", "x": 1, "xanchor": "right"},
			"showlegend": True,
		},
	)
	collapse_legend(figure, mapping)

	return figure


def _edge_traces(edges: pl.DataFrame) -> list[dict]:
	"""One trace per edge, its width and colour set by the edge's ``scaled`` weight.

	The arrowheads ride the figure's colour axis, so Format can swap its scale; a line
	cannot, so its colour is sampled here and resampled by the clientside Format.
	"""
	colorscale: list[str] = sample_colorscale(AURORA, edges["scaled"].to_list())

	return [
		{
			"type": "scatter",
			"x": [edge["x0"], edge["x1"], None],
			"y": [edge["y0"], edge["y1"], None],
			"line": {"width": edge["scaled"] * 10, "color": color},  # scaled up for visibility
			"hoverinfo": "none",
			"mode": "lines+markers",
			"marker": {
				"size": edge["scaled"] * 10,
				"symbol": "arrow",
				"angleref": "previous",
				"color": [edge["scaled"]] * 2,
				"coloraxis": "coloraxis",
			},
			"opacity": 0.5,
			"showlegend": False,
		}
		for edge, color in zip(edges.iter_rows(named=True), colorscale, strict=True)
	]


def _node_trace(
	nodes: pl.DataFrame, colors: list[str], labels: list[str], include_ranking: bool
) -> dict:
	"""One trace holding every node, sized by its ``size`` column.

	Args:
		nodes: one row per node, as ``x``, ``y`` and ``size``.
		colors: one colour per node, aligned with ``nodes``.
		labels: what each node is called, aligned with ``nodes``.
		include_ranking: whether the hover text names the ranking as well as the animal.
	"""
	# A marker needs a positive size; the hover keeps the animal's real value.
	sizes = [size if size > 0 else 0.1 for size in nodes["size"]]

	return {
		"type": "scatter",
		"x": nodes["x"].to_list(),
		"y": nodes["y"].to_list(),
		"text": [f"<b>{label}</b>" for label in labels],
		"hovertext": [
			f"Mouse ID: {label}<br>Ranking: {size}" if include_ranking else f"Mouse ID: {label}"
			for label, size in zip(labels, nodes["size"], strict=True)
		],
		"hoverinfo": "text",
		"mode": "markers+text",
		"textposition": "top center",
		"showlegend": False,
		"marker": {"size": sizes, "color": colors},
	}


def plot_network_graph(
	nodes: pl.DataFrame,
	edges: pl.DataFrame,
	colors: list[str],
	colorway: list[str],
	labels: list[str],
	title: str,
	include_ranking: bool,
) -> Figure:
	"""Plots network graph of social structure.

	Args:
		nodes: laid-out nodes as ``x``, ``y`` and ``size``.
		edges: laid-out edges as ``x0``, ``y0``, ``x1``, ``y1`` and ``scaled`` (0-1).
		colors: one colour per node, aligned with ``nodes``.
		colorway: the colour categories, which the recording page's Format swaps a palette by.
		labels: node names, aligned with ``nodes``.
		title: the figure title.
		include_ranking: whether the hover text names each node's size as its ranking.
	"""
	hidden = {"showticklabels": False, "showgrid": False, "zeroline": False, "automargin": True}

	return Figure(
		data=[*_edge_traces(edges), _node_trace(nodes, colors, labels, include_ranking)],
		layout={
			"showlegend": False,
			"hovermode": "closest",
			"title": {"text": title, "x": 0.5, "y": 0.95},
			"coloraxis": {"colorscale": AURORA, "cmin": 0, "cmax": 1, "showscale": False},
			"xaxis": hidden,
			"yaxis": hidden | {"scaleanchor": "x", "scaleratio": 1},
			"colorway": colorway,
		},
	)


def plot_social_stability(frame: pl.DataFrame, mapping: ColorMapping) -> Figure:
	"""Plots the stability of a social relationship based on time spent together."""
	figure = Figure(
		data=[
			{
				"type": "scatter",
				"name": value,
				"x": _array(rows["stability"]),
				"y": _array(rows["proportion_together"]),
				"customdata": rows.select("animal_id_2").rows(),
				"mode": "markers",
				"marker": {"color": mapping.trace_colors[value], "size": 12},
				"hovertemplate": (
					f"{mapping.trace_column}={value}<br>stability=%{{x}}<br>"
					"proportion_together=%{y}<br>animal_id_2=%{customdata[0]}<extra></extra>"
				),
			}
			for value, rows in _trace_rows(frame, mapping).items()
		],
		layout={
			"title": {"text": "<b>Relationship stability</b>"},
			"xaxis": {"title": {"text": "<b>Relationship stability</b>"}, "range": [0, 1]},
			"yaxis": {"title": {"text": "<b>Median proportion together</b>"}, "range": [0, 1]},
			"legend": {"tracegroupgap": 0},
		},
	)
	collapse_legend(figure, mapping)

	return figure


def plot_quality_heatmap(matrix: np.ndarray, animals: list[str], antennas: list[str]) -> Figure:
	"""Plots the share of each animal's passes over each antenna that went unrecorded."""
	figure = _matrix(
		matrix,
		antennas,
		animals,
		"<b>Missed passes by animal and antenna</b>",
		"Antenna: %{x}<br>Animal: %{y}<br>Missed: %{z:.2f}%<extra></extra>",
	)
	_axis(figure, "xaxis").update(
		title={"text": "<b>Antenna</b>"}, type="category", constraintoward="right"
	)
	_axis(figure, "yaxis").update(automargin=True, title={"text": "<b>Animal ID</b>"})
	figure["layout"]["coloraxis"]["colorbar"] = {"title": {"text": "<b>Missed [%]</b>"}}

	return _crosshair(figure)


def plot_quality_by_antenna(frame: pl.DataFrame) -> Figure:
	"""Plots each antenna's passes as correct, interpolated and bad, stacked to 100%."""
	antennas = frame["antenna"].cast(pl.String).to_list()

	return Figure(
		data=[
			{
				"type": "bar",
				"x": antennas,
				"y": _array(frame[f"{kind}_share"]),
				"customdata": _array(frame[kind]),
				"name": kind.capitalize(),
				"legendgroup": f"reads-{kind}",
				"marker": {"color": READ_LOOKS[kind], "line": {"width": 0}},
				"hovertemplate": (
					f"{kind.capitalize()}: %{{customdata:,}} (%{{y:.2f}}%)<extra></extra>"
				),
			}
			for kind in READ_KINDS
		],
		layout={
			"title": {"text": "<b>Reads per antenna</b>"},
			"barmode": "stack",
			"barcornerradius": CORNER_RADIUS,
			"hovermode": "x unified",
			"xaxis": {"title": {"text": "<b>Antenna</b>"}, "type": "category"},
			"yaxis": {"title": {"text": "<b>Passes [%]</b>"}, "range": [0, 100]},
		},
	)


def plot_cage_preference(
	frame: pl.DataFrame,
	positions: list[str],
	colors: list[str],
	granularity: str,
	value_label: str,
	place: str = "cage",
) -> Figure:
	"""Plots position preference on a per position basis (cohort preference summary)."""
	labels = _tick_labels(positions)
	frame = frame.with_columns(pl.col("position").replace_strict(positions, labels))
	groups = frame.partition_by("position", as_dict=True)
	hover = ["animal_id", granularity, "time_in_position_text"]

	return Figure(
		data=[
			{
				"type": "box",
				"name": label,
				"x": _array(groups[(label,)]["position"]),
				"y": _array(groups[(label,)]["time_in_position"]),
				"customdata": groups[(label,)].select(hover).rows(),
				"marker": {"color": color},
				"boxpoints": "outliers",
				"boxmean": True,
				"hovertemplate": "position=%{x}<br>"
				+ "<br>".join(f"{column}=%{{customdata[{i}]}}" for i, column in enumerate(hover))
				+ "<extra></extra>",
			}
			for label, color in zip(labels, colors, strict=True)
			if (label,) in groups
		],
		layout={
			"title": {"text": f"<b>{place.capitalize()} preference</b>"},
			# Overlay + trace order: a legend-hidden position frees its slot, leaving no gap.
			"boxmode": "overlay",
			"xaxis": {
				"title": {"text": f"<b>{place.capitalize()}s</b>"},
				"categoryorder": "trace",
			},
			"yaxis": {"title": {"text": value_label}},
			"legend": {"title": {"text": "<b>Position</b>"}, "tracegroupgap": 0},
			"colorway": colors,
		},
	)


def plot_timeline(
	frame: pl.DataFrame,
	animals: list[str],
	labels: list[str],
	positions: list[str],
	spans: pl.DataFrame,
	x_range: tuple[str, str],
) -> Figure:
	"""Plot animal positions over time as a compact Gantt-style timeline.

	Args:
		animals: one row per animal, top to bottom.
		labels: what each row is called, aligned with ``animals``.
		x_range: the stretch of wall-clock time the axis shows.
	"""
	colors = dict(zip(positions, sample_palette(len(positions)), strict=True))
	row_dtype = np.min_scalar_type(len(animals))

	groups = frame.partition_by("position", "animal_id", as_dict=True)
	data = []

	# Every position and animal gets a trace, empty or not: plotly keeps a hidden legend
	# entry by trace index, and a zoom redraws the figure with other bars. An empty trace
	# holds one None, since plotly hides a zero-length trace and its legend entry with it.
	for position in positions:
		for row, animal in enumerate(animals):
			rows = groups.get((position, animal), frame.clear())
			x = rows["x"].explode(empty_as_null=False).to_list() or [None]

			data.append(
				{
					"type": "scattergl",
					"x": x,
					# Explicit, not implied: scattergl ignores a y0/dy pair.
					"y": _array(np.full(len(x), row, dtype=row_dtype)),
					"mode": "lines",
					"line": {"width": 10, "color": colors[position]},
					"name": position,
					"legendgroup": position,
					"showlegend": row == 0,
					"meta": animal,
					"hovertemplate": f"{position}<br>Animal: %{{meta}}<br>%{{x}}<extra></extra>",
				}
			)

	figure = Figure(
		data=data,
		layout={
			"title": {"text": "<b>Position timeline</b>"},
			"xaxis": {"title": {"text": "<b>Timeline</b>"}, "type": "date", "range": x_range},
			"yaxis": {
				"tickvals": list(range(len(animals))),
				"ticktext": labels,
				"range": [len(animals) - 0.5, -0.5],
				"zeroline": False,
			},
			"legend": {"title": {"text": "<b>Position</b>"}, "tracegroupgap": 1},
			"colorway": list(colors.values()),
		},
	)
	_event_spans(figure, spans, above=True)

	return figure


def plot_actogram(heatmap: Heatmap, cells: pl.DataFrame, phases: dict[str, float]) -> Figure:
	"""Plots cohort visits as a day-by-hour raster, phase-banded and event-marked."""
	hours = [int(hour) for hour in heatmap.x]
	figure = Figure(
		data=[
			{
				"type": "heatmap",
				"z": _array(heatmap.values[0]),
				"x": hours,
				"y": heatmap.y,
				"colorscale": LAGOON,
				"meta": TILES,
				"hovertemplate": "%{y}, hour %{x}<br>%{z} visits<extra></extra>",
				"colorbar": {**COLORBAR, "title": {"text": heatmap.label, "side": "right"}},
			}
		],
		layout={
			"title": {"text": "<b>Recording pulse</b>"},
			"xaxis": {
				"title": {"text": "<b>Hour since phase onset</b>"},
				"tickvals": [hour - 0.5 for hour in hours[::2]],
				"ticktext": [str(hour) for hour in hours[::2]],
				"showgrid": False,
				"zeroline": False,
			},
			"yaxis": {"autorange": "reversed", "showgrid": False},
		},
	)

	switch = max(phases.values())
	_add_shapes(figure, [_vline(switch - 0.5)])
	_phase_band(figure, phases, switch - 0.5, (hours[0] - 0.5, hours[-1] + 0.5))

	rows = {label: index for index, label in enumerate(heatmap.y)}
	_add_shapes(
		figure,
		[
			{
				"type": "rect",
				"name": "event-span",
				"x0": cell["hour"] - 0.5,
				"x1": cell["hour"] + 0.5,
				"y0": rows[f"Day {cell['day']}"] - 0.5,
				"y1": rows[f"Day {cell['day']}"] + 0.5,
				"line": {"width": 1.5},
				"fillcolor": "rgba(0,0,0,0)",
				"layer": "above",
			}
			for cell in cells.iter_rows(named=True)
		],
	)
	if not cells.is_empty():
		hourly = cells.select(
			"event",
			"index",
			position=pl.lit(None, pl.String),
			x0=pl.col("hour") - 0.5,
			x1=pl.col("hour") + 0.5,
		).unique()
		_event_lanes(figure, _event_runs(hourly), ["x"], 2)

	return figure


def plot_occupancy_ribbon(
	frame: pl.DataFrame,
	order: list[str],
	colors: dict[str, str],
	granularity: str,
	spans: pl.DataFrame,
) -> Figure:
	"""Plots the share of cohort time each place held, stacked to 100% per window unit."""
	data = []
	# A category axis thins its ticks to the width it has; it sits a bin at its index, not
	# its value, so the band and spans are shifted by the first bin.
	first = frame[granularity].min() or 0

	for place, label in zip(order, _tick_labels(order), strict=True):
		rows = frame.filter(pl.col("place") == place).sort(granularity)
		if rows.is_empty():
			continue

		trace = {
			"type": "scatter",
			"x": rows[granularity].to_list(),
			"y": rows["pct"].to_list(),
			"name": label,
			"mode": "lines",
			"stackgroup": "one",
			"line": {"width": 0.5, "color": colors[place], "shape": "spline", "smoothing": 0.8},
			"fillcolor": colors[place],
			"hovertemplate": f"{label}<br>%{{y:.1f}}% of cohort time<extra></extra>",
		}
		if place == Layout.UNDEFINED:
			# Hatched, not another flat band: this is animal-time with no place at all,
			# and it should not read as a fifth cage.
			trace["fillpattern"] = {"shape": "/", "size": 6, "solidity": 0.3}
		data.append(trace)

	figure = Figure(
		data=data,
		layout={
			"title": {"text": "<b>Habitat occupancy</b>"},
			"xaxis": {
				"title": {"text": _X_LABELS[granularity][1]},
				"type": "category",
				"showgrid": False,
			},
			"yaxis": {"title": {"text": "<b>Share of cohort time [%]</b>"}, "range": [0, 100]},
			"hovermode": "x unified",
		},
	)

	if granularity == "phase_count":
		# A phase bin is one phase, so the band names each column rather than marking a
		# single switch - the same two colours the hours slider and the pulse carry.
		bins = frame.select(granularity, "phase").unique().sort(granularity)
		_add_shapes(
			figure,
			[
				_band_segment(
					row["phase"], row[granularity] - first - 0.5, row[granularity] - first + 0.5
				)
				for row in bins.iter_rows(named=True)
			],
		)

	# The bands stack to 100%, so a span behind them would only show through the
	# translucent tunnel and undefined ones at the top.
	_event_spans(figure, spans, above=True, offset=first)

	return figure


def plot_phenotype_map(
	frame: pl.DataFrame, mapping: ColorMapping, label: str = "animal_id"
) -> Figure:
	"""Plots one marker per animal: locomotion, sociality, chases won and rating."""
	# Marker area scales to the largest rating, which draws 30 px across.
	sizeref = frame["rating_size"].max() / 30**2  # ty: ignore[unsupported-operator]
	figure = Figure(
		data=[
			{
				"type": "scatter",
				"name": value,
				"x": _array(rows["visits"]),
				"y": _array(rows["gregariousness"]),
				"text": _array(rows[label]),
				"customdata": rows.select(label, "won", "ordinal").rows(),
				"mode": "markers+text",
				"marker": {
					"color": mapping.trace_colors[value],
					"size": _array(rows["rating_size"]),
					"sizemode": "area",
					"sizeref": sizeref,
					"line": {"width": 1, "color": "rgba(0,0,0,0.25)"},
				},
				"textposition": "top center",
				"textfont": {"size": 10},
				"cliponaxis": False,
				"hovertemplate": (
					"<b>%{customdata[0]}</b><br>%{x} visits<br>"
					"%{y:.0%} of cage time with company<br>"
					"%{customdata[1]} chases won · rating %{customdata[2]:.1f}<extra></extra>"
				),
			}
			for value, rows in _trace_rows(frame, mapping).items()
		],
		layout={
			"title": {"text": "<b>Cohort phenotype map</b>"},
			"xaxis": {"title": {"text": "<b>Visits (locomotion)</b>"}},
			"yaxis": {"title": {"text": "<b>Time in company [%]</b>"}, "tickformat": ".0%"},
			# Legend symbols otherwise take each animal's raw size, shrinking the low-rated ones.
			"legend": {"tracegroupgap": 0, "itemsizing": "constant"},
		},
	)

	# Only a rated animal can top a hierarchy: with no chases there is none to point at.
	rated = frame.drop_nulls("ordinal")
	top = rated.sort("ordinal").row(-1, named=True) if rated.height else None
	if top is not None and top["gregariousness"] is not None:
		active_str = "more" if top["visits"] >= frame["visits"].median() else "less"
		social_str = "more" if top["gregariousness"] >= frame["gregariousness"].median() else "less"

		figure["layout"]["annotations"] = [
			{
				"x": top["visits"],
				"y": top["gregariousness"],
				"ax": -64,
				"ay": 44,
				"text": (
					f"top of the hierarchy:<br>{active_str} active, {social_str} social than median"
				),
				"showarrow": True,
				"arrowhead": 0,
				"arrowwidth": 1,
				"font": {"size": 10},
				"opacity": 0.8,
			}
		]

	collapse_legend(figure, mapping)

	return figure
