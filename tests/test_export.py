"""Tests for plotting.export.fit_for_export - the layout pass, not kaleido rendering.

Rendering through kaleido was checked by hand at 85x64 and 174x140 mm while designing
these rules (see the blueprint); what is pinned here is the geometry the Python port
computes, since that is what a regression could silently get wrong. One test does render,
to pin that the rounded corners the app draws in the browser reach the exported file.
"""

import xml.etree.ElementTree as ET

import numpy as np
import plotly.graph_objects as go
import pytest
from plotly.subplots import make_subplots

from deepecohab.plotting import export


def test_size_and_font_scale_from_mm_and_pt():
	fig = go.Figure(go.Scatter(x=[1], y=[1]))

	fitted, notes = export.fit_for_export(fig, width_mm=85, height_mm=64, pt=8)

	assert fitted.layout.width == round(85 * export.PX_PER_MM)
	assert fitted.layout.height == round(64 * export.PX_PER_MM)
	assert fitted.layout.font.size == pytest.approx(8 * 96 / 72)
	assert notes == []


def test_title_is_drawn_only_when_given():
	fig = go.Figure(go.Scatter(x=[1], y=[1]))

	shown, _ = export.fit_for_export(fig, 85, 64, 8, title="Activity")
	hidden, _ = export.fit_for_export(fig, 85, 64, 8, title="")

	assert shown.layout.title.text == "Activity"
	assert hidden.layout.title.text == ""


def test_legend_always_moves_to_the_side():
	fig = go.Figure(
		[go.Scatter(x=[1], y=[1], name="a"), go.Scatter(x=[1], y=[1], name="b")],
		layout={"showlegend": True},
	)

	fitted, _ = export.fit_for_export(fig, 85, 64, 8)

	assert fitted.layout.legend.orientation == "v"
	assert fitted.layout.legend.x > 1  # past the plot, not underneath it


def test_show_legend_false_drops_it_entirely():
	fig = go.Figure(go.Scatter(x=[1], y=[1], name="a"), layout={"showlegend": True})

	fitted, _ = export.fit_for_export(fig, 85, 64, 8, show_legend=False)

	assert fitted.layout.showlegend is False
	assert fitted.layout.legend.x is None


def test_wide_legend_warns():
	fig = go.Figure(
		[go.Scatter(x=[1], y=[1], name="a very long series name indeed")],
		layout={"showlegend": True},
	)

	_, notes = export.fit_for_export(fig, width_mm=40, height_mm=64, pt=12)

	assert any("legend takes" in note for note in notes)


def test_event_lanes_keep_their_room_when_shown_and_every_mark_drops_otherwise():
	lane = {"type": "rect", "yref": "paper", "ysizemode": "pixel", "yanchor": 1, "y0": 4, "y1": 40}
	fig = go.Figure(
		layout={
			"shapes": [
				{**lane, "name": "event-lane"},
				{"type": "rect", "name": "event-span", "yref": "y domain", "y0": 0, "y1": 1},
				{"type": "line", "name": "phase-switch"},
			],
			"annotations": [{"name": "event-label", "text": "Tone", "yref": "paper", "y": 1.02}],
		}
	)

	shown, _ = export.fit_for_export(fig, 174, 140, 8, show_events=True)
	hidden, _ = export.fit_for_export(fig, 174, 140, 8, show_events=False)

	assert shown.layout.margin.t >= 40
	assert shown.layout.annotations[0].y == 1.02
	assert [shape.name for shape in hidden.layout.shapes] == ["phase-switch"]
	assert hidden.layout.annotations == ()


def test_shown_events_are_unhidden_when_the_dashboard_toggle_hid_them():
	fig = {
		"layout": {
			"shapes": [{"type": "rect", "name": "event-span", "visible": False}],
			"annotations": [{"name": "event-label", "text": "Tone", "visible": False}],
		}
	}

	shown, _ = export.fit_for_export(fig, 174, 140, 8, show_events=True)

	assert shown["layout"]["shapes"][0]["visible"] is True
	assert shown["layout"]["annotations"][0]["visible"] is True
	assert fig["layout"]["shapes"][0]["visible"] is False


def test_the_top_event_lanes_name_is_not_clipped_by_the_margin():
	"""A lane's name is taller than its 6 px bar, so the margin clears the name, not the bar."""
	lane = {"type": "rect", "yref": "paper", "ysizemode": "pixel", "yanchor": 1, "y0": 18, "y1": 24}
	name = {"name": "event-label", "text": "Tone", "yref": "paper", "y": 1, "yshift": 21}
	fig = go.Figure(layout={"shapes": [{**lane, "name": "event-lane"}], "annotations": [name]})

	fitted, _ = export.fit_for_export(fig, 85, 64, 12)

	assert fitted.layout.margin.t >= 21 + 12 * 96 / 72 * export._TICK_SCALE / 2


def _pair_matrices(ids: list[str]) -> go.Figure:
	"""Four square animal-by-animal panels in a row, as the sociability heatmap draws them."""
	fig = make_subplots(rows=1, cols=4, horizontal_spacing=0.06, subplot_titles=list("ABCD"))
	for col in range(1, 5):
		fig.add_trace(go.Heatmap(z=np.ones((len(ids), len(ids))), x=ids, y=ids), row=1, col=col)
		fig.update_yaxes(scaleanchor="x" if col == 1 else f"x{col}", row=1, col=col)
	return fig


def test_square_panels_wrap_into_the_grid_that_draws_them_largest():
	ids = [f"0065-016198{n:04}" for n in range(5)]

	fitted, _ = export.fit_for_export(_pair_matrices(ids), 85, 64, 8)
	wide, _ = export.fit_for_export(_pair_matrices(ids), 180, 64, 8)

	top, bottom = fitted.layout.yaxis.domain, fitted.layout.yaxis3.domain
	assert fitted.layout.yaxis2.domain == top
	assert fitted.layout.yaxis4.domain == bottom
	assert bottom[1] < top[0]
	assert fitted.layout.xaxis3.domain == fitted.layout.xaxis.domain
	titles = {note.text: (note.x, note.y) for note in fitted.layout.annotations}
	assert titles["C"] == pytest.approx((sum(fitted.layout.xaxis3.domain) / 2, bottom[1]), abs=1e-3)
	# The rows already name the columns, and long column labels would eat the panels' height.
	assert not any(fitted.layout[f"xaxis{n}"].showticklabels for n in ("", 2, 3, 4))
	# Wide enough for one row, whose rotated labels fit in the height under it.
	assert wide.layout.yaxis3.domain == (0, 1)
	assert wide.layout.xaxis3.showticklabels


def test_shape_and_scatter_lines_are_capped_at_1_5px():
	fig = go.Figure(
		go.Scatter(x=[1], y=[1], line={"width": 4}),
		layout={"shapes": [{"type": "rect", "line": {"width": 4}}]},
	)

	fitted, _ = export.fit_for_export(fig, 85, 64, 8)

	assert fitted.data[0].line.width == 1.5
	assert fitted.layout.shapes[0].line.width == 1.5


def test_timeline_bars_fill_their_row_instead_of_the_line_cap():
	fig = go.Figure(
		go.Scattergl(x=[1, 2], y=[0, 0], mode="lines", line={"width": 10}),
		layout={"yaxis": {"range": [3.5, -0.5]}},
	)

	small, _ = export.fit_for_export(fig, 85, 64, 8)
	tall, _ = export.fit_for_export(fig, 85, 128, 8)

	assert small.data[0].line.width > export._MAX_LINE_PX
	assert tall.data[0].line.width > 2 * small.data[0].line.width


def test_scattergl_without_a_row_range_takes_the_line_cap():
	fig = go.Figure(go.Scattergl(x=[1, 2], y=[0, 0], mode="lines", line={"width": 10}))

	fitted, _ = export.fit_for_export(fig, 85, 64, 8)

	assert fitted.data[0].line.width == export._MAX_LINE_PX


def test_a_heatmaps_own_colorbar_is_sized_and_cleared_by_the_legend():
	fig = go.Figure(
		[go.Heatmap(z=[[1, 2]], colorbar={"thickness": 30}), go.Scatter(x=[1], y=[1], name="a")],
		layout={"showlegend": True},
	)

	fitted, _ = export.fit_for_export(fig, 85, 64, 8)

	assert fitted.data[0].colorbar.thickness == pytest.approx(
		8 * 96 / 72 * export._COLORBAR_THICKNESS
	)
	assert fitted.data[0].colorbar.tickfont.size == pytest.approx(8 * 96 / 72 * export._TICK_SCALE)
	assert fitted.layout.legend.x > 1.02  # past the colour bar


def test_show_legend_false_drops_colorbars_too():
	fig = go.Figure(
		go.Heatmap(z=[[1, 2]], coloraxis="coloraxis"),
		layout={"coloraxis": {"colorscale": "Viridis"}},
	)
	fig.add_heatmap(z=[[1, 2]])

	fitted, _ = export.fit_for_export(fig, 85, 64, 8, show_legend=False)

	assert fitted.layout.coloraxis.showscale is False
	assert fitted.data[1].showscale is False


def test_a_hidden_legend_takes_no_room_and_raises_no_warning():
	fig = go.Figure(
		[go.Scatter(x=[1], y=[1], name="a very long series name indeed")],
		layout={"showlegend": False},
	)

	fitted, notes = export.fit_for_export(fig, width_mm=40, height_mm=64, pt=12)

	assert notes == []
	assert fitted.layout.legend.x is None


def test_axes_only_a_trace_names_are_fitted_too():
	fig = go.Figure(go.Box(y=[1, 2, 3]))

	fitted, _ = export.fit_for_export(fig, 85, 64, 8)

	assert fitted.layout.xaxis.automargin is True
	assert fitted.layout.yaxis.automargin is True


def test_shapes_above_the_plot_get_room_in_the_top_margin():
	fig = go.Figure(go.Scatter(x=[1], y=[1]))
	bare, _ = export.fit_for_export(fig, 85, 64, 8)
	fig.add_shape(type="rect", xref="x", yref="paper", x0=0, x1=1, y0=1.01, y1=1.035)

	banded, _ = export.fit_for_export(fig, 85, 64, 8)

	assert banded.layout.margin.t > bare.layout.margin.t


def test_polar_labels_get_margins_clear_of_the_legend_even_untitled():
	"""Polar axes have no automargin, so an untitled export used to clip the top label."""
	fig = go.Figure(
		[go.Scatterpolar(r=[1, 2], theta=["Activity", "Long metric"], name="animal")],
		layout={
			"showlegend": True,
			"polar": {"angularaxis": {"labelalias": {"Long metric": "Long<br>metric"}}},
		},
	)

	fitted, _ = export.fit_for_export(fig, 85, 64, 8, title="")

	tick = 8 * 96 / 72 * export._TICK_SCALE
	margin = fitted.layout.margin
	assert margin.l >= export.text_px("Activity", tick)
	assert margin.t == margin.b == pytest.approx(2 * tick * export._LINE_HEIGHT)
	assert margin.r > margin.l  # the labels' room, plus the legend's
	assert (fitted.layout.legend.xref, fitted.layout.legend.x) == ("container", 1)


def test_forced_dtick_reverts_to_automatic_ticking():
	"""_phase_markers forces dtick=1 on the hour axis; export widens it back out."""
	fig = go.Figure(go.Scatter(x=[1], y=[1]), layout={"xaxis": {"dtick": 1}})

	fitted, _ = export.fit_for_export(fig, 85, 64, 8)

	assert fitted.layout.xaxis.dtick is None
	assert fitted.layout.xaxis.tickmode == "auto"


def test_stacked_panels_get_non_overlapping_domains_top_to_bottom():
	fig = go.Figure(
		layout={
			"yaxis": {"domain": [0.55, 1.0]},
			"yaxis2": {"domain": [0.0, 0.45]},
		}
	)

	fitted, _ = export.fit_for_export(fig, 174, 140, 8)

	top = fitted.layout.yaxis.domain
	bottom = fitted.layout.yaxis2.domain
	assert top[0] > bottom[1]  # still stacked, top above bottom, with a gap
	assert top[1] <= 1.0
	assert bottom[0] >= 0.0


def test_too_many_short_panels_warns():
	layout = {f"yaxis{i or ''}": {"domain": [1 - (i + 1) * 0.05, 1 - i * 0.05]} for i in range(10)}

	_, notes = export.fit_for_export(go.Figure(layout=layout), 85, 30, 8)

	assert any("panels get" in note for note in notes)


def test_export_keeps_rounded_boxes_and_tiles(tmp_path):
	if not export.ensure_chrome_available():
		pytest.skip("kaleido has no Chrome to render through")
	fig = make_subplots(rows=1, cols=2)
	fig.add_trace(go.Box(y=[1, 2, 3, 4, 5]), row=1, col=1)
	fig.add_trace(go.Heatmap(z=[[1, 2], [3, 4]], xgap=1, ygap=2), row=1, col=2)
	path = tmp_path / "rounded.svg"

	export.export_figure(fig, path, 85, 64, 8, "svg")

	svg = ET.parse(path).getroot()
	ns = "{http://www.w3.org/2000/svg}"
	box = next(node for node in svg.iter(f"{ns}path") if node.get("class") == "box")
	assert "Q" in box.get("d")
	assert next(svg.iter(f"{ns}image")).get("clip-path", "").startswith("url(#deh-tiles-")


def test_tile_gap_scales_with_the_cells(tmp_path):
	if not export.ensure_chrome_available():
		pytest.skip("kaleido has no Chrome to render through")
	gaps = []
	for size in (60, 120):
		fig = go.Figure(go.Heatmap(z=[[1, 2], [3, 4]], meta="tiles"))
		path = tmp_path / f"tiles{size}.svg"
		export.export_figure(fig, path, size, size, 8, "svg")
		svg = ET.parse(path).getroot()
		rects = [r for r in svg.iter("{http://www.w3.org/2000/svg}rect") if r.get("rx")]
		a, b = sorted(rects, key=lambda r: (float(r.get("y")), float(r.get("x"))))[:2]
		gaps.append(
			(float(b.get("x")) - float(a.get("x")) - float(a.get("width"))) / float(a.get("width"))
		)
	assert gaps[0] > 0
	assert gaps[0] == pytest.approx(gaps[1], rel=0.05)
