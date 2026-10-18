/* Plot behaviour that works on any plotly graph div, redone after every redraw, and the theme
 * repaint, as window.dehRetheme.
 *
 * Loaded by the app as a Dash asset and by the docs site (docs/plots.mjs), which shares it with
 * rounded_corners.js so its interactive figures draw as the dashboard's do. Neither needs Dash.
 */
(function () {
	"use strict";

	/* Square heatmaps shrink their x axis to the cells and push them toward the colour bar
	 * (plot_factory's constraintoward="right"), so a wide card leaves all its spare width left
	 * of the matrix. Sliding the drawing back by half of it centres matrix and colour bar as
	 * one; the figure title, placed on the whole container, is held where it was. */
	function centreSquare(gd) {
		const layout = gd._fullLayout;
		const xaxis = layout && layout.xaxis;
		const paper = gd.querySelector(".svg-container");
		if (!xaxis || !paper) return;
		const domain = xaxis._inputDomain;
		const square = xaxis.constrain === "domain" && xaxis.constraintoward === "right";
		const shift = square ? ((domain[1] - domain[0]) * layout._size.w - xaxis._length) / 2 : 0;
		const slide = (element, px) => element && (element.style.transform = shift > 0.5 ? `translateX(${px}px)` : "");
		slide(paper, -shift);
		slide(gd.querySelector(".g-gtitle"), shift);
	}

	/* A side legend wider than this share of the canvas moves below the plot, where it wraps
	 * instead of squeezing the plot off the canvas. */
	const LEGEND_SIDE_MAX = 1 / 3;
	const LEGEND_BELOW = {orientation: "h", xref: "container", yref: "container", x: 0, xanchor: "left", y: 0, yanchor: "bottom"};

	/* Only a legend the figure left in plotly's default place moves; one a figure pinned stays.
	 * The side width is measured while it is at the side, so a wider canvas can move it back. */
	function fitLegend(gd) {
		const layout = gd._fullLayout;
		const legend = layout && layout.legend;
		if (!legend || !layout.showlegend) return;
		const below = (gd.layout.legend || {}).orientation === "h";
		if (below && !gd._dehLegendW) return;
		if (!below) {
			const placed = gd.layout.legend || {};
			if (placed.x !== undefined || placed.y !== undefined || placed.orientation) return;
			gd._dehLegendW = legend._width;
		}
		const fits = gd._dehLegendW <= LEGEND_SIDE_MAX * layout.width;
		if (below === !fits) return;
		const keys = Object.keys(LEGEND_BELOW);
		const update = Object.fromEntries(keys.map((key) => ["legend." + key, fits ? null : LEGEND_BELOW[key]]));
		if (fits) gd._dehLegendW = 0;
		window.Plotly.relayout(gd, update);
	}

	/* ``layout`` in another theme: the same repaint plot_theme.apply does on the server, from
	 * ``look``, which is plot_theme.looks(name) - the template, and the literal colours of the
	 * phase band and the event outlines, which a template cannot reach. */
	window.dehRetheme = function (layout, look) {
		const repaint = (shape) => {
			const phase = (shape.name || "").replace("phase-band-", "");
			if (shape.name === "event-span") return {...shape, line: {...shape.line, color: look.ink2}};
			return phase in look.bands ? {...shape, fillcolor: look.bands[phase]} : shape;
		};
		return {...layout, template: look.template, shapes: (layout.shapes || []).map(repaint)};
	};

	new MutationObserver(() => {
		document.querySelectorAll(".js-plotly-plot").forEach((gd) => {
			if (gd._dehAfterplot || !gd.on) return;
			gd._dehAfterplot = true;
			gd.on("plotly_afterplot", () => {
				centreSquare(gd);
				fitLegend(gd);
			});
			centreSquare(gd);
			fitLegend(gd);
		});
	}).observe(document, {childList: true, subtree: true});
})();
