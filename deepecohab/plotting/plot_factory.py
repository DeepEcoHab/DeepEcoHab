import math
import textwrap
from typing import Literal

import networkx as nx
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from deepecohab.core.data_model import Layout
from deepecohab.plotting.animals import ColorMapping, collapse_legend
from deepecohab.plotting.export import LEGEND_ITEM_PX, text_px
from deepecohab.plotting.prepare import READ_KINDS, Heatmap
from deepecohab.plotting.theme import (
	AURORA,
	COLORBAR,
	FONT_SIZE,
	PHASE_BAND,
	READ_LOOKS,
	sample_palette,
)


def plot_animal_speed(
	frame: pl.DataFrame,
	positions: list[str],
	colors: list[str],
	granularity: str,
) -> go.Figure:
	"""One box per tunnel, pooling animal/day-or-phase median speeds."""
	figure = px.box(
		frame,
		x="position",
		y="speed_cm_s",
		color="position",
		color_discrete_map=dict(zip(positions, colors, strict=True)),
		category_orders={"position": positions},
		hover_data=["animal_id", granularity],
		points="outliers",
		labels={"speed_cm_s": "Median speed [cm/s]", "position": "Tunnel"},
	)
	figure.update_traces(boxmean=True)
	figure.update_layout(
		title="<b>Tunnel-crossing speed</b>",
		colorway=colors,
		legend={"title": "<b>Tunnel</b>"},
		xaxis_title="<b>Tunnels</b>",
		yaxis_title="<b>Median speed [cm/s]</b>",
	)
	figure.update_xaxes(tickvals=positions, ticktext=_tick_labels(positions))
	return figure


def _tick_labels(names: list[str]) -> list[str]:
	"""Turn snake_case position names into axis tick text."""
	return [name.capitalize().replace("_", " ") for name in names]


def _band_segment(
	figure: go.Figure,
	phase: str,
	x0: float,
	x1: float,
	*,
	xref: str = "x",
	y: tuple[float, float] = (1.01, 1.035),
) -> None:
	"""One stretch of the phase band, in the light theme's colour for ``phase``."""
	colors = PHASE_BAND["light"]

	figure.add_shape(
		type="rect",
		name=f"phase-band-{phase}",
		xref=xref,
		yref="paper",
		x0=x0,
		x1=x1,
		y0=y[0],
		y1=y[1],
		fillcolor=colors["dark_phase"] if phase == "dark_phase" else colors["light_phase"],
		line_width=0,
		layer="above",
	)


def _phase_band(
	figure: go.Figure,
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

	for name, x0, x1 in zip(order, edges[:-1], edges[1:], strict=True):
		_band_segment(figure, name, x0, x1, xref=xref, y=y)


def _phase_markers(figure: go.Figure, phases: dict[str, float]) -> None:
	"""Mark the phase switch on an hour axis that begins at the first phase's onset.

	A single thin line, styled by the active template's shape defaults so it reads
	on either ground without competing with the Phase-coloured traces, and the phase
	band above the panel naming which side is which.
	"""
	switch = max(phases.values())
	hours = [hour for trace in figure.data for hour in (trace.x if trace.x is not None else ())]

	figure.add_vline(x=switch, line_width=1.5)
	# From the data rather than the full day: an hours window narrows these axes.
	if hours:
		_phase_band(figure, phases, switch, (min(hours) - 0.5, max(hours) + 0.5))

	figure.update_layout(xaxis={"dtick": 1})


#: Where a span's label sits inside its box.
_CORNERS: tuple[str, ...] = ("top left", "bottom left", "middle left")


def _pin_bin_axis(figure: go.Figure, bins: pl.Series) -> None:
	"""Hold a binned x axis to its bins, half a bin either side.

	Autorange counts what :func:`_event_spans` draws: a span reaches half a bin past the
	outermost bin, so the axis would run past the data on its own. Pinning the range
	first is also what the bin columns want - a band drawn on the edge bin would
	otherwise push the axis out the same way.
	"""
	figure.update_xaxes(range=[(bins - 0.5).min(), (bins + 0.5).max()])


def _event_spans(
	figure: go.Figure,
	spans: pl.DataFrame,
	facets: list[str] | None = None,
	*,
	outline: bool = False,
) -> None:
	"""Shade where each event falls on a time axis, coloured and labelled by event.

	An event's colour comes from its place among the recording's events, so it is the
	same on every plot. On a faceted figure - one panel per cage - a span with a position
	is drawn only on that cage's panel; without panels it is labelled with its cages
	instead. The faceted figures are heatmaps, where a fill would tint the colour scale,
	so there the spans are outlined - as they are wherever ``outline`` asks for it, which
	is any plot whose traces would otherwise paint over a span drawn behind them.

	The span itself is a shape; its label rides a second, invisible shape named
	``event-label``. A shape's label is clipped to the plot area - an annotation's text
	is not, so a long label ran out over the legend - and this twin is always drawn
	above the data, where a filled span itself has to stay behind it.
	"""
	if spans.is_empty():
		return

	palette = px.colors.qualitative.Pastel
	outline = outline or facets is not None

	if facets is None:
		# With no panel per cage, bouts of one event in different cages share a span.
		spans = spans.group_by("event", "x0", "x1", maintain_order=True).agg(
			pl.col("position").drop_nulls().unique().sort().str.join(", ")
		)
		panels: list[tuple[str, str, str | None]] = [("x", "y", None)]
	else:
		panels = [
			(trace.xaxis, trace.yaxis, facet)
			for facet, trace in zip(facets, figure.data, strict=True)
		]

	# An Enum's physical code is the event's place among the recording's events.
	for span in spans.with_columns(index=pl.col("event").to_physical()).iter_rows(named=True):
		index = span["index"]
		color = palette[index % len(palette)]

		text = span["event"]
		if facets is None and span["position"]:
			text = f"{text} ({span['position']})"

		if outline:
			font = {"size": 10, "color": color}
			shape_style = {"line_color": color, "line_width": 2, "layer": "above"}
		else:
			font = {"size": 10}
			shape_style = {
				"fillcolor": color.replace("rgb", "rgba").replace(")", ", 0.25)"),
				"line_width": 0,
				"layer": "below",
			}
		# Overlapping events would stack their labels, so each event takes its own corner.
		label = {"text": text, "textposition": _CORNERS[index % len(_CORNERS)], "font": font}

		for xaxis, yaxis, facet in panels:
			if facet is not None and span["position"] not in (None, facet):
				continue

			box = {
				"type": "rect",
				"xref": xaxis,
				"yref": f"{yaxis} domain",
				"x0": span["x0"],
				"x1": span["x1"],
				"y0": 0,
				"y1": 1,
			}
			figure.add_shape(name="event-span", **box, **shape_style)
			figure.add_shape(name="event-label", **box, line_width=0, layer="above", label=label)


#: Most panels a faceted heatmap puts in one column (stacked) or one row (grid).
FACETS_PER_LINE = 4


def _crosshair(figure: go.Figure) -> go.Figure:
	"""Trace the hovered cell's row and column across the plot."""
	spikes = {"showspikes": True, "spikemode": "across", "spikethickness": 1, "spikedash": "solid"}
	figure.update_xaxes(**spikes)
	figure.update_yaxes(**spikes)

	return figure


def _faceted_heatmap(
	heatmap: Heatmap,
	title: str,
	x_title: str,
	y_title: str,
	hover: tuple[str, str],
	*,
	square: bool = False,
	grid: bool = False,
) -> go.Figure:
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
	"""
	n = len(heatmap.facets)
	cols = min(n, FACETS_PER_LINE) if grid else math.ceil(n / FACETS_PER_LINE)
	rows = math.ceil(n / cols)
	figure = make_subplots(
		rows=rows,
		cols=cols,
		shared_xaxes=not grid,
		vertical_spacing=min(0.12, 1 / max(rows - 1, 1)),
		horizontal_spacing=0.06,
		subplot_titles=[f"<b>{name.capitalize().replace('_', ' ')}</b>" for name in heatmap.facets],
	)

	value = "%{z}" if heatmap.text is None else "%{customdata}"
	hovertemplate = f"{hover[0]}: %{{x}}<br>{hover[1]}: %{{y}}<br>Value: {value}<extra></extra>"

	for index, values in enumerate(heatmap.values):
		row, col = divmod(index, cols)
		figure.add_trace(
			go.Heatmap(
				z=values,
				x=[str(label) for label in heatmap.x],
				y=heatmap.y,
				customdata=heatmap.text[index] if heatmap.text is not None else None,
				coloraxis="coloraxis",
				hovertemplate=hovertemplate,
			),
			row=row + 1,
			col=col + 1,
		)
		show_x = index + cols >= n
		show_y = col == 0

		figure.update_xaxes(
			showticklabels=show_x,
			ticks="outside" if show_x else "",
			row=row + 1,
			col=col + 1,
		)
		figure.update_yaxes(
			showticklabels=show_y,
			ticks="outside" if show_y else "",
			row=row + 1,
			col=col + 1,
		)

	figure.update_xaxes(automargin=True)
	figure.update_xaxes(title_text=x_title, row=rows, col=1)
	figure.update_yaxes(automargin=True)
	figure.update_yaxes(title_text=y_title, col=1)
	figure.update_layout(
		title=title,
		coloraxis={
			"cmin": 0,
			"colorscale": AURORA,
			"colorbar": {"title": {"text": heatmap.label}},
		},
	)

	if square:
		# Each panel is animal-by-animal; a shared scale keeps its cells square instead
		# of stretched across the panel's width. make_subplots numbers axes in the same
		# row-major order this loop places traces in, so the panel at `index` owns axis
		# `index + 1`.
		for index in range(n):
			row, col = divmod(index, cols)
			axis = index + 1
			figure.update_yaxes(
				scaleanchor="x" if axis == 1 else f"x{axis}", scaleratio=1, row=row + 1, col=col + 1
			)
		# Shrink a panel to its cells, under its title, rather than padding empty rows around it,
		# and toward the colour bar rather than leaving a gap before it.
		figure.update_xaxes(constrain="domain", constraintoward="right")
		figure.update_yaxes(constrain="domain", constraintoward="top")

	return _crosshair(figure)


def _position_plot(
	frame: pl.DataFrame,
	positions: list[str],
	mapping: ColorMapping,
	*,
	y: str,
	title: str,
	x_title: str,
	y_title: str,
	hover: dict[str, bool],
	agg: Literal["sum", "mean"],
	granularity: str,
) -> go.Figure:
	"""Per-animal values against position: grouped bars for a sum, boxes for a mean."""
	# Pinned so the index-based tick labels below name the right category.
	orders = {"position": positions, mapping.trace_column: mapping.order}
	match agg:
		case "sum":
			# px.histogram re-aggregates in the browser and drops customdata with it,
			# so the sum arrives already aggregated and is drawn as plain bars.
			figure = px.bar(
				frame,
				x="position",
				y=y,
				color=mapping.trace_column,
				color_discrete_map=mapping.trace_colors,
				category_orders=orders,
				hover_data=hover,
				title=title,
				barmode="group",
			)
			figure.update_traces(marker_line_width=0)
		case "mean":
			figure = px.box(
				frame,
				x="position",
				y=y,
				color=mapping.trace_column,
				color_discrete_map=mapping.trace_colors,
				category_orders=orders,
				hover_data=hover | {granularity: True},
				title=title,
				boxmode="group",
				points="outliers",
			)
			figure.update_traces(boxmean=True)

	# px pins each trace to an offsetgroup, which reserves a slot even when legend-hidden;
	# without one, plotly groups only the visible traces.
	figure.update_traces(offsetgroup="", alignmentgroup="")
	collapse_legend(figure, mapping)
	figure.update_xaxes(
		title_text=x_title,
		tickvals=list(range(len(positions))),
		ticktext=_tick_labels(positions),
	)
	figure.update_yaxes(title_text=y_title)
	figure.update_layout(barcornerradius=6, legend_title_text=mapping.legend_title)

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
) -> go.Figure:
	"""Plots bar or box graph of cage and tunnel visits or time spent."""
	hover = {mapping.trace_column: True, "position": True, metric: True}

	if metric == "time":
		hover |= {metric: False, "time_text": True}

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
		hover=hover,
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
) -> go.Figure:
	"""Plot time alone per position as a bar or box plot."""
	return _position_plot(
		frame,
		positions,
		mapping,
		y="time_alone",
		title="<b>Time spent alone</b>",
		x_title=f"<b>{place.capitalize()}</b>",
		y_title=value_label,
		hover={
			mapping.trace_column: True,
			"position": True,
			"time_alone": False,
			"time_alone_text": True,
		},
		agg=agg,
		granularity=granularity,
	)


#: Title, y-axis title and hover label of the count line plots, per input.
_LINE_LABELS: dict[str, tuple[str, str, str]] = {
	"activity": ("<b>Activity over time</b>", "<b>Antenna detections</b>", "Detections"),
	"chasings": ("<b>Chasing over time</b>", "<b># of chasing events</b>", "Events"),
	"speed": ("<b>Mean tunnel-crossing speed</b>", "<b>Mean speed [cm/s]</b>", "Speed [cm/s]"),
}

#: Hover label and axis title of each axis a count line plot can run along.
_LINE_X: dict[str, tuple[str, str]] = {
	"hour": ("Hour", "<b>Hour since phase onset</b>"),
	"day": ("Day", "<b>Day</b>"),
	"phase_count": ("Phase", "<b>Phase</b>"),
}


def _line_x_axis(
	figure: go.Figure, bins: pl.Series, phases: dict[str, float], spans: pl.DataFrame
) -> None:
	"""Title and pin a count line's x axis, marking the phase switch on the hour axis only."""
	figure.update_xaxes(title=_LINE_X[bins.name][1])
	if bins.name == "hour":
		_pin_bin_axis(figure, bins)
		_phase_markers(figure, phases)
	else:
		# A category axis ticks whole steps only, widening them as the window grows. It sits
		# a bin at its index among the categories, not its value, and so does a span.
		figure.update_xaxes(type="category", range=[0, bins.n_unique() - 1])
		spans = spans.with_columns(pl.col("x0", "x1") - bins.min())
	_event_spans(figure, spans)


#: Thickness of a side-panel bar or box, in category units.
_SIDE_WIDTH = 0.64


def _side_panel(
	figure: go.Figure,
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
	column = mapping.trace_column
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
	present = set(frame[column].cast(pl.String))
	rows = sorted(
		(row for row in mapping.order if row in present),
		key=lambda row: legend.index(category(row)),
	)

	for row in rows:
		values = frame.filter(pl.col(column) == row)[value]
		side = {
			"xaxis": "x2",
			"yaxis": "y2",
			"name": row,
			"legendgroup": category(row),
			"showlegend": False,
		}
		if agg == "sum":
			trace = go.Bar(
				y=[row],
				x=[values.sum()],
				orientation="h",
				width=_SIDE_WIDTH,
				marker_color=mapping.trace_colors[row],
				hovertemplate=f"{row}: %{{x:,}}<extra></extra>",
				**side,
			)
		else:
			trace = go.Box(
				y=[row] * len(values),
				x=values.to_list(),
				orientation="h",
				boxmean=True,
				width=_SIDE_WIDTH,
				line_color=mapping.trace_colors[row],
				boxpoints="all",
				jitter=0.4,
				pointpos=0,
				marker_size=3,
				**side,
			)
		figure.add_trace(trace)

	lowest_edge, top_edge = len(rows) - 1 + _SIDE_WIDTH / 2, -0.5
	figure.update_layout(
		xaxis_domain=[0, 0.9],
		yaxis_range=[low - pad, high + pad],
		xaxis2={
			"domain": [0.91, 1],
			"anchor": "y2",
			"title": f"<b>{'Total' if agg == 'sum' else 'Mean'} {label.lower()}</b>",
			"labelalias": {"0": ""},
		},
		yaxis2={
			"anchor": "x2",
			"type": "category",
			"showticklabels": False,
			"range": [(lowest_edge - zero_at * top_edge) / (1 - zero_at), top_edge],
		},
		barcornerradius=4,
	)


def plot_sum_line(
	frame: pl.DataFrame,
	mapping: ColorMapping,
	input_type: Literal["activity", "chasings"],
	x: Literal["hour", "day", "phase_count"],
	phases: dict[str, float],
	spans: pl.DataFrame,
) -> go.Figure:
	"""Plots totals for activity or chasings, per hour of the day or per window unit."""
	title, y_axes_label, hover_label = _LINE_LABELS[input_type]

	figure = px.line(
		frame,
		x=x,
		y="total",
		color=mapping.trace_column,
		color_discrete_map=mapping.trace_colors,
		category_orders={mapping.trace_column: mapping.order},
		title=title,
		# px switches to WebGL past 1000 rows, which has no spline.
		render_mode="svg",
	)

	collapse_legend(figure, mapping)
	figure.update_layout(legend={"title": mapping.legend_title})
	figure.update_yaxes(title=y_axes_label)
	figure.update_traces(mode="lines")
	_line_x_axis(figure, frame[x], phases, spans)
	_side_panel(figure, frame, mapping, "sum", hover_label)

	return figure


def plot_mean_line(
	frame: pl.DataFrame,
	mapping: ColorMapping,
	input_type: Literal["activity", "chasings", "speed"],
	x: Literal["hour", "day", "phase_count"],
	phases: dict[str, float],
	spans: pl.DataFrame,
) -> go.Figure:
	"""Plots means with SEM shading, per hour or per window unit."""
	title, y_axes_label, hover_label = _LINE_LABELS[input_type]
	x_label = _LINE_X[x][0]

	figure = go.Figure()

	for trace_value in mapping.order:
		trace_rows = frame.filter(pl.col(mapping.trace_column) == trace_value)
		if trace_rows.is_empty():
			continue
		color = mapping.trace_colors[trace_value]
		category = mapping.category_by_animal.get(trace_value, trace_value)

		bins = trace_rows[x].to_list()
		y = trace_rows["mean"].to_list()
		bounds = trace_rows["upper"].to_list() + trace_rows["lower"].to_list()[::-1]
		shade_color = color.replace("rgb", "rgba").replace(")", ", 0.2)")  # shaded region is SEM

		figure.add_trace(
			go.Scatter(
				x=bins + bins[::-1],
				y=bounds,
				fill="toself",
				fillcolor=shade_color,
				line_color="rgba(255,255,255,0)",
				mode="lines",
				showlegend=False,
				name=trace_value,
				line={"shape": "spline"},
				hoverinfo="skip",
			)
		)

		figure.add_trace(
			go.Scatter(
				x=bins,
				y=y,
				mode="lines",
				line_color=color,
				name=trace_value,
				line={"shape": "spline"},
				hovertemplate=(
					f"{mapping.legend_title}: {category}<br>{x_label}: %{{x}}<br>"
					f"{hover_label}: %{{y:.1f}}<extra></extra>"
				),
			)
		)

	collapse_legend(figure, mapping)
	figure.update_layout(title=title, legend={"title": mapping.legend_title, "tracegroupgap": 0})
	figure.update_yaxes(title=y_axes_label)
	_line_x_axis(figure, frame[x], phases, spans)
	_side_panel(figure, frame, mapping, "mean", hover_label)

	return figure


def plot_ranking_line(frame: pl.DataFrame, mapping: ColorMapping, spans: pl.DataFrame) -> go.Figure:
	"""Plots line graph of ranking over time."""
	figure = px.line(
		# Wall-clock, as the event spans are; not epoch numbers, which plotly would read in
		# the browser's zone.
		frame.with_columns(pl.col("datetime").dt.replace_time_zone(None)),
		x="datetime",
		y="ordinal",
		line_shape="hv",
		color=mapping.trace_column,
		color_discrete_map=mapping.trace_colors,
		category_orders={mapping.trace_column: mapping.order},
	)

	collapse_legend(figure, mapping)
	figure.update_layout(
		title="<b>Social dominance ranking in time</b>",
		legend={"title": mapping.legend_title, "tracegroupgap": 0},
		xaxis={"title": "<b>Timeline</b>", "type": "date"},
		yaxis={"title": "<b>Ranking</b>"},
	)
	_event_spans(figure, spans)

	return figure


def plot_ranking_distribution(frame: pl.DataFrame, mapping: ColorMapping) -> go.Figure:
	"""Plots line graph of ranking distribution with shaded area."""
	figure = px.line(
		frame,
		x="ranking",
		y="probability_density",
		color=mapping.trace_column,
		color_discrete_map=mapping.trace_colors,
		category_orders={mapping.trace_column: mapping.order},
		hover_data=[mapping.trace_column, "ranking", "probability_density"],
	)
	figure.update_traces(fill="tozeroy")

	collapse_legend(figure, mapping)
	figure.update_layout(
		title="<b>Ranking probability distribution</b>",
		xaxis={"title": "<b>Ranking</b>"},
		yaxis={"title": "<b>Probability density</b>"},
		legend={"title": mapping.legend_title, "tracegroupgap": 0},
	)

	return figure


def plot_ranking_stability(
	frame: pl.DataFrame,
	mapping: ColorMapping,
	granularity: str,
	spans: pl.DataFrame,
) -> go.Figure:
	"""Plots animal rank on a per day (or per phase) basis."""
	label = "Phase" if granularity == "phase_count" else "Day"
	cadence = "Daily" if granularity == "day" else "Per-phase"

	figure = go.Figure(
		layout={
			"title_x": 0.5,
			"title": f"<b>{cadence} dominance rank trajectories</b>",
			"legend": {"title": mapping.legend_title, "tracegroupgap": 1},
			"yaxis": {
				"title": "<b>Rank</b>",
				"autorange": "reversed",
				"type": "category",
				"categoryorder": "array",
				"categoryarray": frame["rank"].unique().sort(),
			},
			"xaxis": {"title": f"<b>{label}</b>"},
		}
	)

	for trace_value in mapping.order:
		trace_rows = frame.filter(pl.col(mapping.trace_column) == trace_value).sort(granularity)
		if trace_rows.is_empty():
			continue
		color = mapping.trace_colors[trace_value]

		figure.add_trace(
			go.Scatter(
				x=trace_rows[granularity],
				y=trace_rows["rank"],
				mode="lines+markers",
				name=trace_value,
				line={"color": color},
				marker={"color": color},
			)
		)

	collapse_legend(figure, mapping)
	_event_spans(figure, spans)

	return figure


def plot_time_spent_per_cage(
	heatmap: Heatmap,
	kind: Literal["hourly", "daily"],
	spans: pl.DataFrame,
	place: str = "cage",
	granularity: str = "day",
) -> go.Figure:
	"""Plots one heatmap per position of time spent, by hour or by window unit."""
	match kind:
		case "hourly":
			title = f"<b>Time spent per {place}</b>"
			x_title = "<b>Hour since phase onset</b>"
			hover = ("Hour", "Animal ID")
		case "daily":
			label = "Phase" if granularity == "phase_count" else "Day"
			title = f"<b>{place.capitalize()} preference over time</b>"
			x_title = f"<b>{label}</b>"
			hover = (label, "Animal ID")

	figure = _faceted_heatmap(heatmap, title, x_title, "<b>Animal ID</b>", hover)
	# The x axis is categorical, so a bin sits at its index among the columns, not its value.
	_event_spans(figure, spans.with_columns(pl.col("x0", "x1") - heatmap.x[0]), heatmap.facets)

	return figure


def plot_heatmap(
	matrix: np.ndarray,
	animals: list[str],
	title: str,
	hover: tuple[str, str, str],
) -> go.Figure:
	"""Plots a square animal-by-animal matrix.

	Args:
		matrix: rows and columns in ``animals`` order.
		animals: axis labels for both rows and columns.
		title: figure title.
		hover: hover labels for the column, the row and the value.
	"""
	figure = px.imshow(
		matrix, x=animals, y=animals, zmin=0, color_continuous_scale=AURORA, title=title
	)

	column, row, value = hover
	figure.update_traces(
		hovertemplate=f"{column}: %{{x}}<br>{row}: %{{y}}<br>{value}: %{{z}}<extra></extra>"
	)
	# Square cells leave spare width; it goes left of the matrix, not between it and the colour bar.
	figure.update_layout(
		yaxis={"automargin": True}, xaxis={"automargin": True, "constraintoward": "right"}
	)

	return _crosshair(figure)


def plot_sociability_heatmap(
	heatmap: Heatmap,
	metric: Literal["pairwise_encounters", "time_together"],
) -> go.Figure:
	"""Plots one pair matrix per cage, of meetings or time spent together."""
	title = (
		"<b>Number of pairwise encounters</b>"
		if metric == "pairwise_encounters"
		else "<b>Time spent together</b>"
	)

	return _faceted_heatmap(heatmap, title, "", "", ("X", "Y"), square=True, grid=True)


def plot_metrics_polar(frame: pl.DataFrame, mapping: ColorMapping) -> go.Figure:
	"""Plots mean z-scores of metrics with SEM shading as a polar plot."""
	figure = go.Figure()

	for trace_value in mapping.order:
		group = frame.filter(pl.col(mapping.trace_column) == trace_value)
		if group.is_empty():
			continue
		color = mapping.trace_colors[trace_value]
		shade_color = color.replace("rgb", "rgba").replace(")", ", 0.2)")

		group_closed = pl.concat([group, group.head(1)])
		theta = group_closed["metric"]

		figure.add_trace(
			go.Scatterpolar(
				r=group_closed["lower"],
				theta=theta,
				mode="lines",
				line={"width": 0, "color": color},
				line_shape="spline",
				showlegend=False,
				hoverinfo="skip",
				name=trace_value,
			)
		)

		figure.add_trace(
			go.Scatterpolar(
				r=group_closed["upper"],
				theta=theta,
				mode="lines",
				fill="tonext",
				fillcolor=shade_color,
				line={"width": 0, "color": color},
				line_shape="spline",
				showlegend=False,
				hoverinfo="skip",
				name=trace_value,
			)
		)

		figure.add_trace(
			go.Scatterpolar(
				r=group_closed["mean"],
				theta=theta,
				mode="lines",
				line={"color": color},
				line_shape="spline",
				name=trace_value,
			)
		)

	collapse_legend(figure, mapping)
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
	figure.update_layout(
		title="<b>Animal feature overview</b>",
		title_y=0.95,
		legend_title_text=mapping.legend_title,
		title_x=0.45,
		polar={
			"angularaxis": {"labelalias": aliases},
			"radialaxis": {
				"visible": True,
				# Between the first two metrics, clear of the label at 0°.
				"angle": 180 / max(len(metrics), 1),
				# Series.min/max is typed as a broad union; arithmetic is valid for this column.
				"range": [means.min() - 0.5, means.max() + 0.5],  # ty: ignore[unsupported-operator]
			},
		},
		margin={"l": label_px, "r": label_px + legend_px},
		legend={"tracegroupgap": 0, "xref": "container", "x": 1, "xanchor": "right"},
		showlegend=True,
	)
	figure.update_polars(bgcolor="rgba(0,0,0,0)")

	return figure


def _edge_traces(
	graph: nx.Graph,
	pos: dict[str, np.ndarray],
	edge_weight: Literal["chasings", "proportion_together"] = "chasings",
) -> list[go.Scatter]:
	"""One trace per edge, its width and colour scaled by the edge's weight.

	Weights are z-scored and squashed through a logistic so the palette spans the
	cohort's own range; a cohort whose edges all carry the same weight has no spread
	to scale by, and takes the middle of the colorscale throughout.

	The arrowheads ride the figure's colour axis, so Format can swap its scale; a line
	cannot, so its colour is sampled here and resampled by the clientside Format.

	Args:
		graph: the network, whose edges carry ``edge_weight`` as an attribute.
		pos: node positions, as ``(x, y, ranking)`` per node.
		edge_weight: which edge attribute drives width and colour.

	Returns:
		One :class:`go.Scatter` per edge, self-loops excluded.
	"""
	edge_widths = np.array([graph.edges[edge][edge_weight] for edge in graph.edges()])

	mu: float = edge_widths.mean()
	std: float = edge_widths.std()

	if std == 0 or np.isnan(std):
		normalized_for_colors = np.full_like(edge_widths, 0.5)
	else:
		z_scores = (edge_widths - mu) / std
		normalized_for_colors = 1 / (1 + np.exp(-z_scores))

	colorscale: list[str] = px.colors.sample_colorscale(AURORA, normalized_for_colors.tolist())

	edge_trace: list[go.Scatter] = []

	for index, edge in enumerate(graph.edges()):
		if edge[0] == edge[1]:
			continue
		source_x, source_y = pos[edge[0]][:2]
		target_x, target_y = pos[edge[1]][:2]
		edge_width = normalized_for_colors[index] * 10  # scaled up for visibility

		edge_trace.append(
			go.Scatter(
				x=[source_x, target_x, None],
				y=[source_y, target_y, None],
				line={
					"width": edge_width,
					"color": colorscale[index],
				},
				hoverinfo="none",
				mode="lines+markers",
				marker={
					"size": edge_width,
					"symbol": "arrow",
					"angleref": "previous",
					"color": [normalized_for_colors[index]] * 2,
					"coloraxis": "coloraxis",
				},
				opacity=0.5,
				showlegend=False,
			)
		)

	return edge_trace


def _node_trace(
	pos: dict[str, np.ndarray],
	colors: list[str],
	animals: list[str],
	labels: list[str],
	include_ranking: bool,
) -> go.Scatter:
	"""One trace holding every node, sized by the ranking its position carries.

	Args:
		pos: node positions, as ``(x, y, ranking)`` per node.
		colors: one colour per animal, aligned with ``animals``.
		animals: the nodes to draw, in the order the colours are given.
		labels: what each node is called, aligned with ``animals``.
		include_ranking: whether the hover text names the ranking as well as the animal.
	"""
	sizes = [pos[node][2] if pos[node][2] > 0 else 0.1 for node in animals]

	return go.Scatter(
		x=[pos[node][0] for node in animals],
		y=[pos[node][1] for node in animals],
		text=[f"<b>{label}</b>" for label in labels],
		hovertext=[
			f"Mouse ID: {label}<br>Ranking: {size}" if include_ranking else f"Mouse ID: {label}"
			for label, size in zip(labels, sizes, strict=True)
		],
		hoverinfo="text",
		mode="markers+text",
		textposition="top center",
		showlegend=False,
		marker={"showscale": False, "colorscale": colors, "size": sizes, "color": colors},
	)


def plot_network_graph(
	connections: pl.DataFrame,
	nodes: pl.DataFrame | None,
	animals: list[str],
	colors: list[str],
	graph_type: Literal["chasings", "proportion_together"],
	layout: Literal["spring", "circular"] = "spring",
	edge_cutoff: float = 0,
	labels: list[str] | None = None,
) -> go.Figure:
	"""Plots network graph of social structure.

	Args:
		connections: edge list carrying the weight column for ``graph_type``.
		nodes: ranking table, required for a chasings graph and ignored otherwise.
		animals: every cohort animal, in the order they ring a circular layout.
		colors: one colour per animal, aligned with ``animals``.
		labels: node names aligned with ``animals``, defaulting to the tags themselves.
		graph_type: which relationship the edges carry.
		layout: ``"spring"`` places nodes by edge weight, ``"circular"`` rings them
			evenly so node positions stay comparable between plots.
		edge_cutoff: edges weaker than this percentage of the strongest are dropped
			before the graph is built, so the layout and edge colours are fitted to what
			remains rather than the full network with lines hidden.
	"""
	match graph_type:
		case "chasings":
			assert nodes is not None, "Ranking nodes are required for a chasings network graph."
			edge_weight = "chasings"
			graph_class = nx.DiGraph
			title = "<b>Dominance network graph</b>"
			include_ranking = True
			ordinals = dict(nodes.select("animal_id", "ordinal").iter_rows())
		case "proportion_together":
			edge_weight = "proportion_together"
			graph_class = nx.Graph
			title = "<b>Sociability network graph</b>"
			include_ranking = False
			ordinals = dict.fromkeys(animals, 30)

	connections = connections.filter(
		pl.col(edge_weight) >= pl.col(edge_weight).max() * edge_cutoff / 100
	)
	graph = nx.from_pandas_edgelist(connections, create_using=graph_class, edge_attr=edge_weight)
	graph.add_nodes_from(animals)

	match layout:
		case "spring":
			pos = nx.spring_layout(
				graph, k=0.1, iterations=50, seed=42, weight=edge_weight, method="energy"
			)
		case "circular":
			pos = nx.circular_layout(animals)

	for animal in animals:
		pos[animal] = np.append(pos[animal], ordinals[animal])

	edge_trace = _edge_traces(graph, pos, edge_weight=edge_weight)
	node_trace = _node_trace(pos, colors, animals, labels or animals, include_ranking)

	figure = go.Figure(
		data=[*edge_trace, node_trace],
		layout=go.Layout(
			showlegend=False,
			hovermode="closest",
			title={"text": title, "x": 0.5, "y": 0.95},
			coloraxis={"colorscale": AURORA, "cmin": 0, "cmax": 1, "showscale": False},
		),
	)

	figure.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, automargin=True)
	figure.update_yaxes(
		showticklabels=False,
		showgrid=False,
		zeroline=False,
		automargin=True,
		scaleanchor="x",
		scaleratio=1,
	)

	return figure


def plot_social_stability(frame: pl.DataFrame, mapping: ColorMapping) -> go.Figure:
	"""Plots the stability of a social relationship based on time spent together."""
	figure = px.scatter(
		frame,
		x="stability",
		y="proportion_together",
		color=mapping.trace_column,
		color_discrete_map=mapping.trace_colors,
		category_orders={mapping.trace_column: mapping.order},
		hover_data={"animal_id_2": True},
		range_x=[0, 1],
		range_y=[0, 1],
		title="<b>Relationship stability</b>",
	)

	collapse_legend(figure, mapping)
	figure.update_layout(
		xaxis={"title": "<b>Relationship stability</b>"},
		yaxis={"title": "<b>Median proportion together</b>"},
		legend_title_text=mapping.legend_title,
	)
	figure.update_traces(marker_size=12)

	return figure


def plot_quality_heatmap(matrix: np.ndarray, animals: list[str], antennas: list[str]) -> go.Figure:
	"""Plots the share of each animal's passes over each antenna that went unrecorded."""
	figure = px.imshow(
		matrix,
		x=antennas,
		y=animals,
		zmin=0,
		color_continuous_scale=AURORA,
		title="<b>Missed passes by animal and antenna</b>",
	)
	figure.update_traces(
		hovertemplate="Antenna: %{x}<br>Animal: %{y}<br>Missed: %{z:.2f}%<extra></extra>"
	)
	figure.update_layout(
		xaxis={"title": "<b>Antenna</b>", "type": "category", "constraintoward": "right"},
		yaxis={"automargin": True, "title": "<b>Animal ID</b>"},
		coloraxis_colorbar={"title": {"text": "<b>Missed [%]</b>"}},
	)

	return _crosshair(figure)


def plot_quality_by_antenna(frame: pl.DataFrame) -> go.Figure:
	"""Plots each antenna's passes as correct, interpolated and bad, stacked to 100%."""
	antennas = frame["antenna"].cast(pl.String).to_list()
	figure = go.Figure(
		[
			go.Bar(
				x=antennas,
				y=frame[f"{kind}_share"],
				customdata=frame[kind],
				name=kind.capitalize(),
				legendgroup=f"reads-{kind}",
				marker={"color": READ_LOOKS[kind], "line": {"width": 0}},
				hovertemplate=(
					f"{kind.capitalize()}: %{{customdata:,}} (%{{y:.2f}}%)<extra></extra>"
				),
			)
			for kind in READ_KINDS
		]
	)
	figure.update_layout(
		barmode="stack", barcornerradius=10, hovermode="x unified", title="<b>Reads per antenna</b>"
	)
	figure.update_xaxes(title_text="<b>Antenna</b>", type="category")
	figure.update_yaxes(title_text="<b>Passes [%]</b>", range=[0, 100])

	return figure


def plot_cage_preference(
	frame: pl.DataFrame,
	positions: list[str],
	colors: list[str],
	granularity: str,
	value_label: str,
	place: str = "cage",
) -> go.Figure:
	"""Plots position preference on a per position basis (cohort preference summary)."""
	labels = _tick_labels(positions)
	frame = frame.with_columns(pl.col("position").replace_strict(positions, labels))
	positions = labels
	figure = px.box(
		frame,
		x="position",
		y="time_in_position",
		color="position",
		points="outliers",
		hover_data={
			"animal_id": True,
			granularity: True,
			"time_in_position": False,
			"time_in_position_text": True,
		},
		color_discrete_map=dict(zip(positions, colors, strict=True)),
		category_orders={"position": positions},
		title=f"<b>{place.capitalize()} preference</b>",
	)

	figure.update_traces(boxmean=True)
	# Overlay + trace order: a legend-hidden position frees its slot instead of leaving a gap.
	figure.update_layout(colorway=colors, boxmode="overlay", legend={"title": "<b>Position</b>"})
	figure.update_yaxes(title_text=value_label)
	figure.update_xaxes(title_text=f"<b>{place.capitalize()}s</b>", categoryorder="trace")

	return figure


def plot_timeline(
	frame: pl.DataFrame, animals: list[str], positions: list[str], spans: pl.DataFrame
) -> go.Figure:
	"""Plot animal positions over time as a compact Gantt-style timeline."""
	colors = dict(zip(positions, sample_palette(len(positions)), strict=True))
	row_dtype = np.min_scalar_type(len(animals))

	# Wall-clock strings, not epoch numbers: plotly reads a number on a date axis in the
	# browser's zone, shifting every bar by its UTC offset. None breaks the line between bars.
	visits = frame.select(
		"position",
		"animal_id",
		x=pl.concat_list(
			pl.col("start").dt.replace_time_zone(None).dt.to_string(),
			pl.col("end").dt.replace_time_zone(None).dt.to_string(),
			pl.lit(None, pl.String),
		),
	)
	groups = visits.partition_by("position", "animal_id", as_dict=True)
	figure = go.Figure()

	# Every position and animal gets a trace, empty or not: plotly keeps a hidden legend
	# entry by trace index, and a zoom redraws the figure with other bars. An empty trace
	# holds one None, since plotly hides a zero-length trace and its legend entry with it.
	for position in positions:
		for row, animal in enumerate(animals):
			rows = groups.get((position, animal), visits.clear())
			x = rows["x"].explode(empty_as_null=False).to_list() or [None]

			figure.add_trace(
				go.Scattergl(
					x=x,
					y=np.full(len(x), row, dtype=row_dtype),
					mode="lines",
					line={"width": 10, "color": colors[position]},
					name=position,
					legendgroup=position,
					showlegend=row == 0,
					meta=animal,
					hovertemplate=f"{position}<br>Animal: %{{meta}}<br>%{{x}}<extra></extra>",
				)
			)

	figure.update_yaxes(
		title=None,
		tickvals=list(range(len(animals))),
		ticktext=animals,
		range=[len(animals) - 0.5, -0.5],
		zeroline=False,
	)
	figure.update_xaxes(title="<b>Timeline</b>", type="date")
	figure.update_layout(
		title="<b>Position timeline</b>",
		legend={"title": "<b>Position</b>", "tracegroupgap": 1},
		colorway=list(colors.values()),
	)

	_event_spans(figure, spans)
	return figure


def plot_actogram(heatmap: Heatmap, cells: pl.DataFrame, phases: dict[str, float]) -> go.Figure:
	"""Plots cohort visits as a day-by-hour raster, phase-banded and event-marked."""
	hours = [int(hour) for hour in heatmap.x]
	figure = go.Figure(
		go.Heatmap(
			z=heatmap.values[0],
			x=hours,
			y=heatmap.y,
			colorscale=AURORA,
			xgap=2,
			# A fixed 2 px gap outgrows the cell once months of days share the card's height.
			# ponytail: row count stands in for pixels; use the drawn height if cards resize.
			ygap=2 if len(heatmap.y) <= 36 else 0,
			hovertemplate="%{y}, hour %{x}<br>%{z} visits<extra></extra>",
			colorbar={**COLORBAR, "title": {"text": heatmap.label, "side": "right"}},
		)
	)

	switch = max(phases.values())
	figure.add_vline(x=switch - 0.5, line_width=1.5)
	_phase_band(figure, phases, switch - 0.5, (hours[0] - 0.5, hours[-1] + 0.5))

	palette = px.colors.qualitative.Pastel
	rows = {label: index for index, label in enumerate(heatmap.y)}
	for cell in cells.iter_rows(named=True):
		row = rows.get(f"Day {cell['day']}")
		if row is None or cell["hour"] not in hours:
			continue

		figure.add_shape(
			type="rect",
			name="event-span",
			x0=cell["hour"] - 0.5,
			x1=cell["hour"] + 0.5,
			y0=row - 0.5,
			y1=row + 0.5,
			line={"width": 1.5, "color": palette[cell["index"] % len(palette)]},
			fillcolor="rgba(0,0,0,0)",
			layer="above",
		)

	figure.update_layout(
		title="<b>Recording pulse</b>",
		xaxis={
			"title": {"text": "<b>Hour since phase onset</b>"},
			"tickvals": [hour - 0.5 for hour in hours[::2]],
			"ticktext": [str(hour) for hour in hours[::2]],
			"showgrid": False,
			"zeroline": False,
		},
		yaxis={"autorange": "reversed", "showgrid": False},
	)

	return figure


def plot_occupancy_ribbon(
	frame: pl.DataFrame,
	order: list[str],
	colors: dict[str, str],
	granularity: str,
	spans: pl.DataFrame,
) -> go.Figure:
	"""Plots the share of cohort time each place held, stacked to 100% per window unit."""
	figure = go.Figure()
	# A category axis thins its ticks to the width it has; it sits a bin at its index, not
	# its value, so the band and spans are shifted by the first bin.
	first = frame[granularity].min() or 0

	for place in order:
		rows = frame.filter(pl.col("place") == place).sort(granularity)
		if rows.is_empty():
			continue

		label = place.capitalize().replace("_", " ")
		trace = go.Scatter(
			x=rows[granularity].to_list(),
			y=rows["pct"].to_list(),
			name=label,
			mode="lines",
			stackgroup="one",
			line={"width": 0.5, "color": colors[place], "shape": "spline", "smoothing": 0.8},
			fillcolor=colors[place],
			hovertemplate=f"{label}<br>%{{y:.1f}}% of cohort time<extra></extra>",
		)
		if place == Layout.UNDEFINED:
			# Hatched, not another flat band: this is animal-time with no place at all,
			# and it should not read as a fifth cage.
			trace.update(fillpattern={"shape": "/", "size": 6, "solidity": 0.3})
		figure.add_trace(trace)

	if granularity == "phase_count":
		# A phase bin is one phase, so the band names each column rather than marking a
		# single switch - the same two colours the hours slider and the pulse carry.
		bins = frame.select(granularity, "phase").unique().sort(granularity)
		for row in bins.iter_rows(named=True):
			x = row[granularity] - first
			_band_segment(figure, row["phase"], x - 0.5, x + 0.5)

	# The bands stack to 100%, so a span behind them would only show through the
	# translucent tunnel and undefined ones at the top.
	_event_spans(figure, spans.with_columns(pl.col("x0", "x1") - first), outline=True)
	label = "Phase" if granularity == "phase_count" else "Day"
	figure.update_layout(
		title="<b>Habitat occupancy</b>",
		xaxis={"title": {"text": f"<b>{label}</b>"}, "type": "category", "showgrid": False},
		yaxis={"title": {"text": "<b>Share of cohort time [%]</b>"}, "range": [0, 100]},
		hovermode="x unified",
	)

	return figure


def plot_phenotype_map(
	frame: pl.DataFrame, mapping: ColorMapping, label: str = "animal_id"
) -> go.Figure:
	"""Plots one marker per animal: locomotion, sociality, chases won and rating."""
	# A rating's zero is arbitrary (mu - 3 sigma, often negative), so area grows from the
	# cohort's lowest rating, offset so that animal still gets a ninth of the top area.
	rating = pl.col("ordinal")
	span = rating.max() - rating.min()
	frame = frame.with_columns(
		pl.when(span > 0).then(rating - rating.min() + span / 8).otherwise(1).alias("rating_size")
	)
	fig = px.scatter(
		frame,
		x="visits",
		y="gregariousness",
		size="rating_size",
		size_max=30,
		color=mapping.trace_column,
		color_discrete_map=mapping.trace_colors,
		category_orders={mapping.trace_column: mapping.order},
		text=label,
		custom_data=[label, "won", "ordinal"],
		labels={
			"visits": "<b>Visits (locomotion)</b>",
			"gregariousness": "<b>Time in company [%]</b>",
			"ordinal": "<b>Dominance</b>",
		},
		title="<b>Cohort phenotype map</b>",
	)

	# Apply fine-grained trace styling
	fig.update_traces(
		textposition="top center",
		textfont_size=10,
		cliponaxis=False,
		marker_line={"width": 1, "color": "rgba(0,0,0,0.25)"},
		hovertemplate=(
			"<b>%{customdata[0]}</b><br>%{x} visits<br>"
			"%{y:.0%} of cage time with company<br>"
			"%{customdata[1]} chases won · rating %{customdata[2]:.1f}<extra></extra>"
		),
	)

	# Dynamic annotation for top animal
	top = frame.sort("ordinal", nulls_last=True).row(-1, named=True)
	if top["gregariousness"] is not None:
		active_str = "more" if top["visits"] >= frame["visits"].median() else "less"
		social_str = "more" if top["gregariousness"] >= frame["gregariousness"].median() else "less"

		fig.add_annotation(
			x=top["visits"],
			y=top["gregariousness"],
			ax=-64,
			ay=44,
			text=(f"top of the hierarchy:<br>{active_str} active, {social_str} social than median"),
			showarrow=True,
			arrowhead=0,
			arrowwidth=1,
			font={"size": 10},
			opacity=0.8,
		)

	# Formatting adjustments
	collapse_legend(fig, mapping)
	# Legend symbols otherwise take each animal's raw size, shrinking the low-rated ones.
	fig.update_layout(
		yaxis_tickformat=".0%", legend={"title": mapping.legend_title, "itemsizing": "constant"}
	)

	return fig
