import copy
import math
import re
import textwrap
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Literal, overload

import plotly.graph_objects as go
import plotly.io as pio

PX_PER_MM = 96 / 25.4

_ROUNDED_CORNERS_JS = Path(__file__).parents[1] / "app" / "assets" / "rounded_corners.js"

_CHAR = 0.55
"""Average glyph width as a fraction of font size, for text-fit estimates."""

_MIN_PANEL_LINES = 3.5
"""Panels shorter than this many lines of text at the chosen point size are warned about."""

_TICK_SCALE = 0.875
"""Tick and legend font size, as a fraction of the base font size."""

_TITLE_SCALE = 1.15
"""Figure title font size, as a multiple of the base font size."""

_LINE_HEIGHT = 1.6
"""One line of text, as a multiple of its font size: the height a title or legend row needs."""

_ROW_TITLE_HEIGHT = 1.7
"""The gap a facet row title needs above its panel, as a multiple of the base font size."""

_PANEL_GAP = 0.9
"""The gap between untitled stacked panels, as a multiple of the base font size."""

_MAX_PANEL_GAP = 0.3
"""Total share of the plot height the inter-panel gaps may take."""

_ABOVE_PANEL = 0.08
"""How far above a panel, in paper units, an annotation still belongs to that panel."""

_MIN_TICK_GAP = 1.15
"""Smallest slot a tick label may sit in, as a multiple of the tick font size."""

_MAX_LINE_PX = 1.5
"""Line width is capped here: a heavier stroke reads as a smear at export sizes."""

_BAR_FILL = 0.6
"""Share of its row a timeline bar fills, close to its 10 px in the app's 340 px panel."""

_EVENT_SHAPES = ("event-span", "event-lane")
"""Shapes an event draws: its mark over the data and its lane above the plot."""

LEGEND_ITEM_PX = 46
"""Swatch, padding and margin around a legend label, in pixels."""

_LEGEND_WARN_FRACTION = 0.4
"""A legend wider than this share of the figure earns a warning."""

_PLOT_WIDTH_FRACTION = 0.72
"""Share of the figure width the plotting area gets once axes and margins are paid for."""

_COLORBAR_THICKNESS = 0.8
"""Colour bar thickness, as a multiple of the base font size, floored at 6 px."""

_AXIS_STANDOFF = {"y": 0.4, "x": 0.3}
"""Gap between an axis title and its ticks, as a multiple of the base font size."""

_TICK_SPACING = 3.2
"""Smallest gap between automatic x ticks, as a multiple of the tick font size."""

_TICK_ROW_HEIGHT = 1.5
"""Height a row of tick labels needs, as a multiple of the tick font size."""

_COLORBAR_LABEL_CHARS = 4
"""Tick labels beside a colour bar, in characters: how far past it the legend must sit."""

_WIDE_LABEL_SCALE = 2.5
"""Labels longer than this many tick heights are thinned rather than rotated."""

_ROTATE_BELOW = 1.1
"""Horizontal labels are rotated once their slot falls below this share of their width."""

_ROTATED_TICK_GAP = 1.3
"""Smallest slot a rotated tick label may sit in, as a multiple of the tick font size."""

_TAG_RE = re.compile(r"<[^>]+>")


def _plain_text(text: Any) -> str:
	"""Strip HTML tags a plotly title, tick label or annotation text may carry."""
	return _TAG_RE.sub("", str(text or ""))


def text_px(text: Any, size: float) -> float:
	"""Estimated width of the widest ``<br>``-separated line of ``text`` at font ``size``."""
	return max(len(_plain_text(line)) for line in str(text or "").split("<br>")) * _CHAR * size


def _title_text(title: Any) -> str:
	"""The text of a plotly title, given either as a bare string or a title dict."""
	if isinstance(title, str):
		return title
	if isinstance(title, dict):
		return str(title.get("text") or "")
	return ""


def _wrap_text(text: str, max_chars: int) -> str:
	"""Greedy word wrap to roughly ``max_chars`` per line, joined with plotly's ``<br>``."""
	# Neither break keeps a word whole past max_chars, as a plot title needs.
	words = " ".join(text.split())
	return "<br>".join(
		textwrap.wrap(words, max_chars, break_long_words=False, break_on_hyphens=False)
	)


def _axis_keys(layout: dict[str, Any], letter: str) -> list[str]:
	"""Every ``{letter}axis[N]`` key in ``layout``, in axis order (``xaxis`` first)."""
	prefix = f"{letter}axis"
	pattern = re.compile(rf"^{re.escape(prefix)}\d*$")
	keys = [key for key in layout if pattern.match(key)]
	return sorted(keys, key=lambda key: int(key[len(prefix) :] or 1))


def _category_labels(data: list[dict[str, Any]], key: str, letter: str) -> list[str] | None:
	"""String tick labels for one axis, read off a heatmap trace that uses it.

	A category axis carries no tick labels of its own in plotly - the labels live on
	the trace - so thinning them has to look at the trace, not the axis.
	"""
	prefix = f"{letter}axis"
	ref = letter + key[len(prefix) :]

	for trace in data:
		if trace.get("type") != "heatmap":
			continue
		if (trace.get(prefix) or letter) != ref:
			continue
		values = trace.get(letter)
		if isinstance(values, list) and values and isinstance(values[0], str):
			return values

	return None


def _thin_axis(axis: dict[str, Any], labels: list[Any], slot: float, tick: float) -> None:
	"""Drop enough tick labels that the survivors fit their slot, rotating first."""
	longest = max(len(str(label)) for label in labels) * _CHAR * tick

	if longest <= tick * _WIDE_LABEL_SCALE:
		step = max(1, math.ceil((longest * _LINE_HEIGHT) / slot))
	else:
		axis["tickangle"] = -90 if slot < longest * _ROTATE_BELOW else 0
		step = max(1, math.ceil((tick * _ROTATED_TICK_GAP) / slot))

	if step > 1:
		kept = labels[::step]
		axis.update({"tickmode": "array", "tickvals": kept, "ticktext": kept})


@dataclass(frozen=True)
class _Size:
	"""The export's pixel size and font sizes, which every layout pass reads."""

	width: int
	height: int
	pt: float
	base: float
	tick: float

	@classmethod
	def of(cls, width_mm: float, height_mm: float, pt: float) -> "_Size":
		base = pt * 96 / 72
		return cls(
			round(width_mm * PX_PER_MM), round(height_mm * PX_PER_MM), pt, base, base * _TICK_SCALE
		)


_Role = Literal["event", "column_title", "row_title", "panel_title", "other"]


def _role(note: dict[str, Any]) -> _Role:
	"""What an annotation is, read off the figure as it was drawn - before any pass moves it.

	``column_title`` sits above the plot (a facet column's label), ``row_title`` is a
	rotated facet row label beside it, and ``panel_title`` already sits on top of its panel.
	"""
	on_paper = (note.get("yref") or "paper") == "paper"
	if note.get("name") == "event-label":
		return "event"
	if on_paper and note.get("y", 0) > 1.001 and note.get("yanchor") != "bottom":
		return "column_title"
	if (note.get("xref") or "paper") == "paper" and abs(note.get("textangle") or 0) == 90:
		return "row_title"
	if on_paper and note.get("yanchor") == "bottom":
		return "panel_title"
	return "other"


def _domain(axis: dict[str, Any]) -> tuple[float, float]:
	lo, hi = axis.get("domain", [0, 1])
	return lo, hi


@cache
def _draws_cartesian_axes(trace_type: str) -> bool:
	return "xaxis" in go.Figure(data=[{"type": trace_type}]).data[0]


def _name_implicit_axes(data: list[dict[str, Any]], layout: dict[str, Any]) -> None:
	"""Name every axis a trace draws in ``layout``, so the axis passes reach them all.

	A trace draws its axes whether or not the layout names them.
	"""
	for trace in data:
		if _draws_cartesian_axes(trace.get("type", "scatter")):
			for letter in "xy":
				layout.setdefault(f"{letter}axis{(trace.get(f'{letter}axis') or letter)[1:]}", {})


def _base_layout(layout: dict[str, Any], size: _Size, title: str, show_legend: bool) -> None:
	layout.update(
		{
			"width": size.width,
			"height": size.height,
			"autosize": False,
			"showlegend": layout.get("showlegend") if show_legend else False,
			"font": {**layout.get("font", {}), "size": size.base},
			"margin": {"l": 2, "r": 2, "t": 2, "b": 2, "pad": 0, "autoexpand": True},
			"title": {
				"text": title or "",
				"font": {"size": size.base * _TITLE_SCALE},
				"automargin": True,
				"x": 0,
				"xanchor": "left",
				"yref": "container",
				"y": 0.995,
				"yanchor": "top",
			},
		}
	)


def _drop_events(layout: dict[str, Any]) -> None:
	layout["shapes"] = [s for s in layout["shapes"] if s.get("name") not in _EVENT_SHAPES]
	layout["annotations"] = [n for n in layout["annotations"] if n.get("name") != "event-label"]


def _show_events(layout: dict[str, Any]) -> None:
	"""Unhide event marks the dashboard's Events toggle left in the figure, hidden."""
	for item in (*layout["shapes"], *layout["annotations"]):
		if item.get("name") in (*_EVENT_SHAPES, "event-label"):
			item["visible"] = True


def _place_annotations(roles: list[tuple[_Role, dict[str, Any]]], size: _Size) -> None:
	"""Scale every annotation, and move facet titles to the top-left of their panel.

	Event names ride their lane, pixels above the plot, so they neither move nor rescale.
	"""
	for role, note in roles:
		note["font"] = {**note.get("font", {}), "size": size.tick if role == "event" else size.base}
		if role == "column_title":
			note.update({"y": 1, "yanchor": "bottom", "yshift": 1})
		elif role == "row_title":
			note.update(
				{
					"text": _plain_text(note.get("text")).replace("_", " "),
					"x": 0,
					"xanchor": "left",
					"xshift": 0,
					"textangle": 0,
					"yanchor": "bottom",
				}
			)


def _legend_entries(data: list[dict[str, Any]], show_legend: bool) -> list[str]:
	"""The legend's entries, one per legend group, as they would be drawn."""
	entries: dict[str, str] = {}
	if show_legend:
		for trace in data:
			if trace.get("showlegend") is False or not trace.get("name"):
				continue
			entries.setdefault(
				trace.get("legendgroup") or trace["name"], _plain_text(trace["name"])
			)
	return list(entries.values())


def _legend_px(entries: list[str], size: _Size) -> float:
	"""Width the legend takes beside the plot, or 0 without one."""
	if not entries:
		return 0
	return max(len(name) for name in entries) * _CHAR * size.tick + LEGEND_ITEM_PX


def _place_legend(
	layout: dict[str, Any],
	entries: list[str],
	size: _Size,
	room: tuple[float, float, float],
	has_colorbar: bool,
) -> list[str]:
	"""Pin the legend at the side, past the colour bar when there is one.

	Args:
		room: the plot height, and the top and bottom margins, the legend has to fit in.

	Returns:
		Warnings about a legend too wide or too tall to fit.
	"""
	if not entries:
		return []

	plot_h, top_px, bottom_px = room
	beside = (size.base * _COLORBAR_THICKNESS + size.tick * _COLORBAR_LABEL_CHARS) / (
		size.width * _PLOT_WIDTH_FRACTION
	)
	layout["legend"] = {
		"font": {"size": size.tick},
		"title": {"text": ""},
		"tracegroupgap": 0,
		"bgcolor": "rgba(0,0,0,0)",
		"itemwidth": 30,
		"orientation": "v",
		"x": 1.02 + beside if has_colorbar else 1.01,
		"xanchor": "left",
		"y": 1,
		"yanchor": "top",
	}

	notes = []
	item_px = _legend_px(entries, size)
	if item_px > _LEGEND_WARN_FRACTION * size.width:
		notes.append(
			f"The legend takes {round(100 * item_px / size.width)}% of the width; "
			"widen the figure or shorten the names."
		)
	legend_h = len(entries) * size.tick * _LINE_HEIGHT
	if legend_h > plot_h:
		needed = math.ceil((legend_h + top_px + bottom_px) / PX_PER_MM)
		notes.append(
			f"The legend needs {needed} mm of height for {len(entries)} entries at {size.pt} pt."
		)
	return notes


def _style_colorbars(
	layout: dict[str, Any], data: list[dict[str, Any]], size: _Size, show_legend: bool
) -> bool:
	"""Size every colour bar for print: a colour axis's, or a heatmap's own.

	A colour bar is the heatmap's legend, so dropping the legend drops it too.

	Returns:
		Whether any colour bar is drawn, which the legend has to clear.
	"""
	owners = [layout[key] for key in layout if key.startswith("coloraxis")]
	owners += [t for t in data if t.get("type") == "heatmap" and not t.get("coloraxis")]
	for owner in owners:
		if not show_legend:
			owner["showscale"] = False
		bar = owner.get("colorbar") or {}
		owner["colorbar"] = {
			**bar,
			# Print wants a physical thickness, not the theme's fraction of the plot.
			"thicknessmode": "pixels",
			"thickness": max(6, size.base * _COLORBAR_THICKNESS),
			"outlinewidth": 0,
			"tickfont": {"size": size.tick},
			"title": {**(bar.get("title") or {}), "font": {"size": size.tick}, "side": "right"},
			"xpad": 4,
			"ypad": 0,
		}
	return any(owner.get("showscale") is not False for owner in owners)


def _vertical_room(
	layout: dict[str, Any], size: _Size, has_title: bool, has_titles: bool
) -> tuple[float, float, float]:
	"""The plot height, and the top and bottom margins around it.

	Args:
		has_title: a figure title is drawn.
		has_titles: panels carry titles above them.
	"""
	top_px = (size.base * _LINE_HEIGHT if has_title else 0) + (
		size.base * _ROW_TITLE_HEIGHT if has_titles else 0
	)
	bottom_px = size.tick * _TICK_ROW_HEIGHT + size.base * _LINE_HEIGHT

	# Event lanes are sized in pixels above the phase band, so their room is added as is - up
	# to the top of the highest lane's name, which stands taller than its bar.
	shapes = layout["shapes"]
	pixel = [shape for shape in shapes if shape.get("ysizemode") == "pixel"]
	names = [note for note in layout["annotations"] if note.get("name") == "event-label"]
	top_px += max(
		[max(shape["y0"], shape["y1"]) for shape in pixel]
		+ [note.get("yshift", 0) + size.tick * _LINE_HEIGHT / 2 for note in names],
		default=0,
	)

	# A shape in paper units above the plot - the phase band - sits in the top margin, and
	# overhangs by a share of the plot height, so its room comes out of that height too.
	overhang = max(
		(
			max(shape.get("y0", 0), shape.get("y1", 0)) - 1
			for shape in shapes
			if shape.get("yref") == "paper" and shape.get("ysizemode") != "pixel"
		),
		default=0,
	)
	if overhang > 0:
		top_px += overhang * (size.height - top_px - bottom_px) / (1 + overhang)

	return max(size.height - top_px - bottom_px, 1), top_px, bottom_px


def _fit_square_panels(
	layout: dict[str, Any],
	data: list[dict[str, Any]],
	roles: list[tuple[_Role, dict[str, Any]]],
	size: _Size,
	plot: tuple[float, float],
) -> None:
	"""Lay square panels out in the grid that draws them largest in the ``plot`` size.

	A square panel is only as big as the shorter side of its cell, so a row of them in a
	figure less wide than the row is long leaves the height below it empty - and rotated
	column labels eat what height there is. A pair matrix's columns are its rows, named
	beside it in the same order, so its column labels go wherever they cost panel size.
	"""
	keys = [
		(f"xaxis{layout[key]['anchor'][1:]}", key)
		for key in _axis_keys(layout, "y")
		if layout[key].get("scaleanchor") and layout[key].get("anchor")
	]
	if not keys:
		return
	plot_w, plot_h = plot
	panels = [(layout[x], layout[y]) for x, y in keys]
	n = len(panels)
	columns = sorted({_domain(x) for x, _ in panels})
	gap = columns[1][0] - columns[0][1] if len(columns) > 1 else 0
	title_px = size.base * _ROW_TITLE_HEIGHT

	def side(cols: int, label_px: float) -> float:
		rows = math.ceil(n / cols)
		return min(
			plot_w * (1 - gap * (cols - 1)) / cols,
			(plot_h - label_px - (rows - 1) * title_px) / rows,
		)

	# ponytail: assumes long column labels always rotate and every panel is n-by-n square.
	labels = _category_labels(data, keys[0][0], "x")
	longest = max(text_px(label, size.tick) for label in labels) if labels else 0
	label_px = longest if longest > size.tick * _WIDE_LABEL_SCALE else 0
	shown = labels != _category_labels(data, keys[0][1], "y") or max(
		side(c, label_px) for c in range(1, n + 1)
	) >= max(side(c, 0) for c in range(1, n + 1))
	cols = max(range(1, n + 1), key=lambda c: side(c, label_px if shown else 0))

	if cols != len(columns):
		width = (1 - gap * (cols - 1)) / cols
		rows = math.ceil(n / cols)
		# Rows that touched would leave a title on the boundary to the panel above it.
		row_gap = title_px / plot_h
		height = (1 - row_gap * (rows - 1)) / rows
		titles = [note for role, note in roles if role == "panel_title"]
		for index, (x, y) in enumerate(panels):
			(x_lo, x_hi), (_, y_hi) = _domain(x), _domain(y)
			row, col = divmod(index, cols)
			x["domain"] = [col * (width + gap), col * (width + gap) + width]
			y["domain"] = [1 - row * (height + row_gap) - height, 1 - row * (height + row_gap)]
			y.update({"showticklabels": col == 0, "ticks": "outside" if col == 0 else ""})
			x.update({"showticklabels": index + cols >= n})
			for note in titles:
				if x_lo <= note.get("x", -1) <= x_hi and abs(note.get("y", -1) - y_hi) < 1e-3:
					note.update({"x": sum(x["domain"]) / 2, "y": y["domain"][1]})
					titles.remove(note)
					break
	for x, _ in panels:
		x["showticklabels"] = shown and x.get("showticklabels", True)
		x["ticks"] = "outside" if x["showticklabels"] else ""


def _stack_panels(
	layout: dict[str, Any],
	roles: list[tuple[_Role, dict[str, Any]]],
	size: _Size,
	room: tuple[float, float, float],
	titled: bool,
) -> list[str]:
	"""Re-space stacked panels for gaps sized to their titles, carrying annotations along.

	Args:
		room: the plot height, and the top and bottom margins around it.
		titled: panels carry row titles, so the gaps between them must fit one.

	Returns:
		A warning when the panels come out too short to read.
	"""
	y_keys = _axis_keys(layout, "y")

	def rounded(key: str) -> tuple[float, float]:
		lo, hi = _domain(layout[key])
		return (round(lo, 4), round(hi, 4))

	rows = sorted(dict.fromkeys(rounded(key) for key in y_keys), key=lambda d: -d[1])
	n = len(rows)
	if n < 2:
		return []

	plot_h, top_px, bottom_px = room
	gap_px = size.base * (_ROW_TITLE_HEIGHT if titled else _PANEL_GAP)
	gap = min(gap_px / plot_h, _MAX_PANEL_GAP / (n - 1))
	span = (1 - gap * (n - 1)) / n
	mapping = {
		row: (max(0.0, 1 - (i + 1) * span - i * gap), min(1.0, 1 - i * span - i * gap))
		for i, row in enumerate(rows)
	}
	old_domains = {key: rounded(key) for key in y_keys}
	for key in y_keys:
		layout[key]["domain"] = list(mapping[old_domains[key]])

	for role, note in roles:
		if (note.get("yref") or "paper") != "paper" or note.get("y") is None:
			continue
		if role in ("column_title", "event"):
			continue
		for (lo, hi), (new_lo, new_hi) in mapping.items():
			if lo - 1e-3 <= note["y"] <= hi + _ABOVE_PANEL:
				scale = (new_hi - new_lo) / (hi - lo)
				if role == "row_title":
					note["y"] = new_hi
				elif note["y"] <= hi:
					note["y"] = new_lo + (note["y"] - lo) * scale
				else:
					note["y"] = new_hi + (note["y"] - hi) * scale
				break

	if span * plot_h >= size.base * _MIN_PANEL_LINES:
		return []
	needed = math.ceil(
		(n * size.base * _MIN_PANEL_LINES + (n - 1) * gap_px + top_px + bottom_px) / PX_PER_MM
	)
	return [
		(
			f"{n} panels get {round(span * plot_h / PX_PER_MM)} mm each; give it at least "
			f"{needed} mm of height at {size.pt} pt."
		)
	]


def _style_axis(
	axis: dict[str, Any], size: _Size, length_px: float, min_chars: int, standoff: float
) -> None:
	"""Automargin, tick font and a title wrapped to the ``length_px`` the axis spans."""
	title = axis.get("title")
	text = _plain_text(_title_text(title))
	axis.update(
		{
			"automargin": True,
			"tickfont": {**axis.get("tickfont", {}), "size": size.tick},
			"title": {
				**(title if isinstance(title, dict) else {}),
				"text": _wrap_text(text, max(min_chars, int(length_px / (_CHAR * size.base))))
				if text
				else "",
				"standoff": standoff,
				"font": {"size": size.base},
			},
		}
	)


def _fit_cartesian_axes(
	layout: dict[str, Any], data: list[dict[str, Any]], size: _Size, plot_h: float, plot_w: float
) -> None:
	"""Type every axis and thin dense category ticks; a forced x ``dtick`` goes automatic."""
	for key in _axis_keys(layout, "y"):
		axis = layout[key]
		lo, hi = _domain(axis)
		px_h = (hi - lo) * plot_h
		_style_axis(axis, size, px_h, 8, size.base * _AXIS_STANDOFF["y"])
		labels = _category_labels(data, key, "y")
		if labels and px_h / len(labels) < size.tick * _MIN_TICK_GAP:
			step = math.ceil((size.tick * _MIN_TICK_GAP) / (px_h / len(labels)))
			kept = labels[::step]
			axis.update({"tickmode": "array", "tickvals": kept, "ticktext": kept})

	for key in _axis_keys(layout, "x"):
		axis = layout[key]
		_style_axis(axis, size, plot_w, 10, size.base * _AXIS_STANDOFF["x"])
		labels = _category_labels(data, key, "x")
		if labels:
			lo, hi = _domain(axis)
			_thin_axis(axis, labels, (hi - lo) * plot_w / len(labels), size.tick)
		elif axis.get("dtick") is not None:
			axis.pop("dtick", None)
			axis.update(
				{"tickmode": "auto", "nticks": max(3, int(plot_w / (size.tick * _TICK_SPACING)))}
			)


def _fit_polar(
	layout: dict[str, Any], data: list[dict[str, Any]], size: _Size, top_px: float, legend_px: float
) -> None:
	"""Margins that clear the angular labels, and a legend pinned beyond them.

	Polar axes have no automargin: the angular labels hang past the circle.
	"""
	if "polar" not in layout:
		return
	polar = layout["polar"]
	angular = polar.get("angularaxis") or {}
	polar["angularaxis"] = {**angular, "tickfont": {"size": size.tick}}
	polar["radialaxis"] = {**(polar.get("radialaxis") or {}), "tickfont": {"size": size.tick}}

	aliases = angular.get("labelalias") or {}
	labels = [
		str(aliases.get(t, t))
		for trace in data
		if trace.get("theta") is not None
		for t in trace["theta"]
	]
	if not labels:
		return
	label_px = max(text_px(label, size.tick) for label in labels) + size.tick
	label_h = max(label.count("<br>") + 1 for label in labels) * size.tick * _LINE_HEIGHT
	layout["margin"].update(
		{"l": label_px, "r": label_px + legend_px, "t": top_px + label_h, "b": label_h}
	)
	if legend_px:
		layout["legend"].update({"xref": "container", "x": 1, "xanchor": "right"})


def _cap_lines(layout: dict[str, Any], data: list[dict[str, Any]], plot_h: float) -> None:
	for shape in layout["shapes"]:
		line = shape.get("line")
		if line and line.get("width"):
			line["width"] = min(line["width"], _MAX_LINE_PX)

	for trace in data:
		line = trace.get("line")
		if trace.get("type") not in ("scatter", "scattergl") or not (line and line.get("width")):
			continue
		axis = (
			layout[f"yaxis{(trace.get('yaxis') or 'y')[1:]}"]
			if trace["type"] == "scattergl"
			else {}
		)
		if axis.get("range"):
			# Only the position timeline draws with scattergl, and its lines are bars - one
			# row per y unit - so they fill a share of their row rather than take the cap.
			lo, hi = _domain(axis)
			line["width"] = (
				_BAR_FILL * (hi - lo) * plot_h / abs(axis["range"][0] - axis["range"][1])
			)
		else:
			line["width"] = min(line["width"], _MAX_LINE_PX)


def styled(figure: dict[str, Any], style: str) -> dict[str, Any]:
	"""``figure`` drawn in the export dialog's ``style``: its own theme, or ``publication``."""
	if style != "publication":
		return figure
	template = pio.templates["publication"].to_plotly_json()
	return {**figure, "layout": {**figure.get("layout", {}), "template": template}}


@overload
def fit_for_export(
	figure: go.Figure,
	width_mm: float,
	height_mm: float,
	pt: float,
	title: str = "",
	show_legend: bool = True,
	show_events: bool = True,
) -> tuple[go.Figure, list[str]]: ...


@overload
def fit_for_export(
	figure: dict[str, Any],
	width_mm: float,
	height_mm: float,
	pt: float,
	title: str = "",
	show_legend: bool = True,
	show_events: bool = True,
) -> tuple[dict[str, Any], list[str]]: ...


def fit_for_export(
	figure: go.Figure | dict[str, Any],
	width_mm: float,
	height_mm: float,
	pt: float,
	title: str = "",
	show_legend: bool = True,
	show_events: bool = True,
) -> tuple[go.Figure | dict[str, Any], list[str]]:
	"""Lay a figure out for its physical export size before kaleido renders it.

	The rules, in order: one font scale from ``pt``; automargin on every axis and the
	title, with long axis titles wrapped to the space they have; stacked panels get
	gaps sized to their titles, and row titles keep their spot; the legend always sits
	at the side, past the colour bar when there is one; dense category ticks are
	thinned, and a forced ``dtick`` goes back to automatic; event lanes keep their room
	above the plot and their names take the tick size, or every event mark is dropped
	when ``show_events`` is false; lines are capped at 1.5 px.

	Args:
		figure: the figure to export, as a ``go.Figure`` or the dict a ``dcc.Graph``
			carries; it is left untouched. Its ``layout.template`` is used as given -
			callers choose ``light``, ``dark`` or ``publication`` before calling this.
		width_mm: physical width.
		height_mm: physical height.
		pt: base font size in points; every other size scales from it.
		title: figure title to draw, or ``""`` to omit it.
		show_legend: keep the legend, or drop it entirely.
		show_events: keep the event lanes, names and marks, or drop them.

	Returns:
		A new figure sized and typeset for export, of the same kind as ``figure`` - a
		dict skips plotly's validation, which costs far more than the layout pass - and
		warnings about anything that still would not fit - an oversized legend, or panels
		too short to read.
	"""
	source = figure.to_dict() if isinstance(figure, go.Figure) else copy.deepcopy(figure)
	data: list[dict[str, Any]] = source.get("data", [])
	layout: dict[str, Any] = source.get("layout", {})
	layout["shapes"] = layout.get("shapes") or []
	layout["annotations"] = layout.get("annotations") or []
	size = _Size.of(width_mm, height_mm, pt)

	_name_implicit_axes(data, layout)
	_base_layout(layout, size, title, show_legend)
	if show_events:
		_show_events(layout)
	else:
		_drop_events(layout)

	roles = [(_role(note), note) for note in layout["annotations"]]
	titled = any(role in ("row_title", "panel_title") for role, _ in roles)
	_place_annotations(roles, size)
	has_colorbar = _style_colorbars(layout, data, size, show_legend)

	has_titles = titled or any(role == "column_title" for role, _ in roles)
	room = _vertical_room(layout, size, bool(title), has_titles)
	plot_h, top_px, _ = room
	entries = _legend_entries(data, layout["showlegend"] is not False)
	legend_px = _legend_px(entries, size)
	plot_w = size.width * _PLOT_WIDTH_FRACTION - legend_px
	notes = _place_legend(layout, entries, size, room, has_colorbar)
	_fit_square_panels(layout, data, roles, size, (plot_w, plot_h))
	notes += _stack_panels(layout, roles, size, room, titled)
	layout["margin"]["t"] = max(2, top_px)
	# Event names are row labels in the left gutter, which automargin doesn't count.
	names = [note for role, note in roles if role == "event"]
	if names:
		layout["margin"]["l"] = max(
			text_px(note["text"], size.tick) - 2 * note.get("xshift", 0) for note in names
		)

	_fit_cartesian_axes(layout, data, size, plot_h, plot_w)
	_fit_polar(layout, data, size, top_px, legend_px)
	_cap_lines(layout, data, plot_h)

	fitted = {**source, "data": data, "layout": layout}
	return (go.Figure(fitted) if isinstance(figure, go.Figure) else fitted), notes


@cache
def ensure_chrome_available() -> bool:
	"""Whether kaleido has a Chrome binary to render through, checked once per process.

	kaleido 1.3 needs one; call this at app start so export can be disabled with a
	clear message instead of every attempt failing deep inside kaleido.
	"""
	import kaleido

	try:
		kaleido.get_chrome_sync()
		return True
	except Exception:
		return False


def export_figure(
	figure: go.Figure | dict[str, Any],
	path: str | Path,
	width_mm: float,
	height_mm: float,
	pt: float,
	format: Literal["svg", "pdf", "png"],
	dpi: int = 300,
	title: str = "",
	show_legend: bool = True,
	show_events: bool = True,
) -> list[str]:
	"""Fit a figure to its export size and render it to disk.

	SVG and PDF keep the physical size the figure was fit to; a PNG is rasterised at
	``scale = dpi / 96`` so its pixel dimensions match the requested resolution while
	the layout - built for ``width_mm``/``height_mm`` at 96 px/inch - stays unscaled.

	Args:
		figure: the figure to export.
		path: where to write the file.
		width_mm: physical width.
		height_mm: physical height.
		pt: base font size in points.
		format: image format kaleido renders. No TIFF or EPS: kaleido 1 dropped EPS.
		dpi: raster resolution; ignored for ``svg`` and ``pdf``.
		title: figure title to draw, or ``""`` to omit it.
		show_legend: keep the legend, or drop it entirely.
		show_events: keep the event lanes, names and marks, or drop them.

	Returns:
		Warnings from :func:`fit_for_export`, so a caller can surface them alongside
		the file it wrote.
	"""
	import kaleido

	fitted, notes = fit_for_export(figure, width_mm, height_mm, pt, title, show_legend, show_events)
	scale = dpi / 96 if format == "png" else 1
	# Rounded boxes and tiles are drawn by the app's script, not by plotly, so kaleido's page
	# loads it too - which figure.write_image has no way to ask for.
	page = kaleido.PageGenerator(others=[(_ROUNDED_CORNERS_JS.as_uri(), "utf-8")])
	kaleido.write_fig_sync(
		fitted, path=path, opts={"format": format, "scale": scale}, kopts={"page_generator": page}
	)

	return notes
