"""Tests for the recording-page callbacks that moved into ``assets/clientside.js``.

Each one replaced a server callback, so the checks run the JS in node and compare it against
the Python it was ported from, such as the export filename against the string ops the old
callback did.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from dash import Dash
from plotly.io.json import to_json_plotly

Dash(__name__, use_pages=True, pages_folder="")

CLIENTSIDE_JS = Path(__file__).parent.parent / "deepecohab" / "app" / "assets" / "clientside.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs node")

_PRELUDE = """
// The renderer defines these two; the asset file only adds its own namespace.
const PREVENT = {description: "Throw to prevent updating all Outputs."};
global.window = {
	dash_clientside: {PreventUpdate: PREVENT, no_update: {}, set_props: () => {}},
	addEventListener: () => {},
};
// The heatmap centring watches the page, which node does not have.
global.document = {body: {}, addEventListener: () => {}};
global.MutationObserver = class { observe() {} };
eval(require("fs").readFileSync(process.argv[2], "utf8"));
const deh = window.dash_clientside.deh;
const wanted = JSON.parse(require("fs").readFileSync(process.argv[3], "utf8"));

const eq = (got, want, what) => {
	if (JSON.stringify(got) !== JSON.stringify(want)) {
		throw new Error(`${what}: ${JSON.stringify(got)} !== ${JSON.stringify(want)}`);
	}
};
const prevents = (fn, what) => {
	try { fn(); } catch (err) { if (err === PREVENT) return; throw err; }
	throw new Error(what + ": expected PreventUpdate");
};
const fire = (triggered_id, value) => {
	window.dash_clientside.callback_context = {
		triggered_id,
		triggered: [{value}],
	};
};
"""

_HARNESS = (
	_PRELUDE
	+ """
// --- openExport: the right figure and title, out of several ---------------
const ids = [{plot: "activity-bar"}, {plot: "ranking-line"}, {plot: "chasings-heatmap"}];
const figures = ids.map((id, i) => ({data: [{y: [i]}], layout: {}}));
// deliberately a different order, as Dash gives no ordering guarantee between two ALL states
const titleIds = [{plot: "chasings-heatmap"}, {plot: "activity-bar"}, {plot: "ranking-line"}];
const titles = ["Chasings heatmap", "Activity per position", "Social dominance ranking"];

// in yet another order: a request that has not been filled in yet is only {tab}
const requestIds = [{plot: "ranking-line"}, {plot: "activity-bar"}, {plot: "chasings-heatmap"}];
const filled = {context: {recording: "Cohort 1 WT"}, opts: {}};
const requests = [{tab: "social"}, filled, {tab: "social"}];
const exportOf = (plot, context) => {
	fire({type: "card-export", plot: plot}, 1);
	const args = [null, figures, ids, titles, titleIds, context, requests, requestIds, null];
	return deh.openExport(...args);
};

eq(
	exportOf("activity-bar", {recording: "Cohort 1 WT"}),
	[
		true,
		{
			figure: figures[0],
			title: "Activity per position",
			data: {plot: "activity-bar", request: filled, x_range: null},
		},
		wanted.filename,
	],
	"openExport carries the figure, title, filename and the request that drew the plot"
);
eq(exportOf("ranking-line", null)[1].data, null, "a card not drawn from a request has no data");
eq(
	exportOf("activity-bar", null)[2],
	"recording__activity-bar",
	"openExport falls back when there is no recording context"
);

const zoomedSpan = ["2023-05-24 01:00", "2023-05-24 02:00"];
const timelineIds = [{plot: "recording-timeline"}];
const exportedZoom = (zoom) => {
	fire({type: "card-export", plot: "recording-timeline"}, 1);
	const figure = {data: [], layout: {xaxis: {range: zoomedSpan}}};
	const request = {context: {recording: "r"}, opts: {}};
	const args = [null, [figure], timelineIds, [], [], null, [request], timelineIds, zoom];
	return deh.openExport(...args)[1].data.x_range;
};
eq(exportedZoom(zoomedSpan), zoomedSpan, "a zoomed timeline exports the span it shows");
eq(exportedZoom(["a", "b"]), null, "a zoom the timeline was since redrawn without is left out");

// --- switchTab: the rec-tab mirror plotRequest reads (rec-tabs itself is built into rec-body,
// too late to be an Input), with the tab written into the address bar alone, not url.search.
const replaced = [];
const linked = [];
global.history = {state: null, replaceState: (_state, _title, href) => replaced.push(href)};
window.dash_clientside.set_props = (id, props) => linked.push([id.index, props.href]);
const switchTo = (tab, search) => {
	global.location = {pathname: "/recording", search};
	replaced.length = linked.length = 0;
	return deh.switchTab(tab);
};
eq(switchTo("social", "?project=abc&tab=overview"), "social", "switchTab mirrors the tab");
eq(replaced, ["/recording?project=abc&tab=social"], "switchTab rewrites only the tab");
eq(linked, [["/recording", "/recording?project=abc&tab=social"]], "the sidebar link follows");
switchTo("social", "?project=abc");
eq(replaced, ["/recording?project=abc&tab=social"], "switchTab adds tab");
// The tab the body mounts with is already in the address bar; the mirror still has to be filled.
eq(switchTo("social", "?tab=social"), "social", "switchTab on the current tab");
eq([replaced, linked], [[], []], "the current tab leaves the address bar alone");
window.dash_clientside.set_props = () => {};

// --- windowControl: bounds, marks and the reset on a granularity change ----------------
for (const [granularity, bound] of [["day", wanted.days], ["phase_count", wanted.phases]]) {
	fire("rec-granularity", granularity);
	const bounds = {days: wanted.days, phases: wanted.phases};
	const out = deh.windowControl(granularity, [2, 3], bounds, {color_by: "sex"});
	eq(out[0], 1, "window min");
	eq(out[1], bound, "window max");
	eq(out[3], [1, bound], "a granularity change re-spans the window");
	eq(out[4], {color_by: "sex", granularity, window: [1, bound]}, "controls merged");
}

// every mark up to 12, then every other one, always keeping the last
const marks = (bound) => {
	fire("rec-granularity", "day");
	return deh.windowControl("day", [1, 1], {days: bound, phases: 0}, null)[2];
};
eq(marks(1), [{value: 1, label: "1"}], "one mark");
eq(marks(12).map((m) => m.value), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "12 marks");
eq(marks(13).map((m) => m.value), [1, 2, 4, 6, 8, 10, 12, 13], "13 thins");
eq(marks(30).map((m) => m.value).slice(-3), [20, 25, 30], "the last mark is kept");
eq(marks(108).map((m) => m.value), [1, 20, 40, 60, 80, 108], "100 yields to the end");
eq(marks(215).map((m) => m.value), [1, 50, 100, 150, 215], "three digits thin further");

fire("rec-window", [2, 3]);
eq(
	deh.windowControl("day", [2, 3], {days: wanted.days, phases: wanted.phases}, null)[3],
	[2, 3],
	"a drag keeps the dragged window"
);

// the readout above the slider follows the window
const painted = {};
window.dash_clientside.set_props = (id, props) => (painted[id] = props);
fire("rec-window", [2, 3]);
deh.windowControl("day", [2, 3], {days: 5, phases: 10}, null);
eq(
	[painted["rec-window-label"].children, painted["rec-window-hint"].children],
	["Days 2 → 3", "2 of 5"],
	"window readout"
);
fire("rec-granularity", "phase_count");
deh.windowControl("phase_count", [2, 3], {days: 5, phases: 10}, null);
eq(painted["rec-window-hint"].children, "all 10", "a whole window reads all");
prevents(() => deh.windowControl("day", [2, 3], null, null), "windowControl without a context");

// --- filterControls: a dict merge, plus the group-mean disable rule -------------------
// The slider hands over hour boundaries [6, 19]; the controls keep the bins 6..18.
eq(
	deh.filterControls([6, 19], null, "sex", "subject_name", true, {window: [1, 5]}),
	[
		{
			window: [1, 5],
			hours: [6, 18],
			phases: [],
			color_by: "sex",
			label_by: "subject_name",
			group_mean: true,
		},
		false,
		{},
		{},
	],
	"filterControls merges into the controls it was given and enables group mean"
);
eq(
	deh.filterControls([6, 19], null, "animal_id", "animal_id", true, {window: [1, 5]}),
	[
		{
			window: [1, 5],
			hours: [6, 18],
			phases: [],
			color_by: "animal_id",
			label_by: "animal_id",
			group_mean: false,
		},
		true,
		{},
		{},
	],
	"filterControls forces group mean off and disables the switch when colouring by animal"
);

// the hours band splits under the other onset, whichever phase the hours count from
const hours = (bounds, phases, context) =>
	deh.filterControls(bounds, phases, "animal_id", "animal_id", false, {}, context);
hours([0, 24], ["dark_phase"], {
	onsets: {light_phase: "07:00", dark_phase: "19:30"},
	start_from: "dark_phase",
});
eq(
	[painted["rec-hours-label"].children, painted["rec-hours-hint"].children],
	["Hours 0 → 24", "whole day"],
	"hours readout"
);
const band = painted["rec-hours-band"].style.background;
eq(band.includes(" 0 47.916666666666664%"), true, "band splits at 11.5 h");
hours([2, 5], [], {onsets: {light_phase: "07:00"}, start_from: "light_phase"});
eq(painted["rec-hours-hint"].children, "3 of 24 h", "a narrowed hours hint");
window.dash_clientside.set_props = () => {};

// the phase chips and the hours slider move each other
const twoPhases = {onsets: {light_phase: "07:00", dark_phase: "19:00"}, start_from: "light_phase"};
fire("rec-phases");
const byPhase = hours([0, 24], ["dark_phase"], twoPhases);
eq(
	[byPhase[0].hours, byPhase[2], byPhase[3]],
	[[12, 23], [12, 24], {}],
	"a phase chip sets the hours"
);
fire("rec-hours");
const byHours = hours([0, 6], ["light_phase", "dark_phase"], twoPhases);
eq(
	[byHours[0].phases, byHours[2], byHours[3]],
	[["light_phase"], {}, ["light_phase"]],
	"the hours set the phase chips"
);

// --- toggleEvents: sets only the figures whose event items show the wrong way --------
const sets = [];
window.dash_clientside.set_props = (id, props) => sets.push([id, props]);
const withEvents = {
	data: [{y: [1]}],
	layout: {
		shapes: [{name: "event-span", visible: true}, {name: "other", visible: true}],
		annotations: [{name: "event-label", visible: true}, {name: "other", visible: true}],
	},
};
const noEvents = {data: [], layout: {shapes: [{name: "other"}]}};
const plots = ["a", "b", "c", "d"].map((plot) => ({plot}));
deh.toggleEvents(false, [withEvents, {}, null, noEvents], plots);
eq(sets.map(([id]) => id.plot), ["a"], "toggleEvents sets only the figure that has events");
const hidden = sets[0][1].figure;
eq(
	hidden.layout.shapes,
	[{name: "event-span", visible: false}, {name: "other", visible: true}],
	"toggleEvents hides only event-span shapes"
);
eq(
	hidden.layout.annotations,
	[{name: "event-label", visible: false}, {name: "other", visible: true}],
	"toggleEvents hides only event-label annotations"
);
sets.length = 0;
deh.toggleEvents(false, [hidden], [{plot: "a"}]);
eq(sets.length, 0, "a figure already hiding its events is left alone");
deh.toggleEvents(true, [hidden], [{plot: "a"}]);
eq(sets[0][1].figure.layout.shapes[0].visible, true, "toggleEvents shows event-span shapes again");
sets.length = 0;
deh.toggleEvents(true, [withEvents], [{plot: "a"}]);
eq(sets.length, 0, "a figure already showing its events is left alone");

// --- plotRequest: only the showing tab asks, and only for inputs it was not drawn with -
const NO = window.dash_clientside.no_update;
const metric = (value) => [{id: {plot: "bar", option: "metric"}, value}];
const full = {
	window: [1, 2],
	granularity: "day",
	hours: [0, 23],
	phases: ["light_phase", "dark_phase"],
	group_mean: false,
};
const context = {recording: "r", days: 2, phases: 4};
const request = (
	tab, stores, options = metric("time"), controls = full, badges = [], trigger, hover
) => {
	window.dash_clientside.callback_context = {
		triggered_id: trigger,
		inputs_list: [null, null, null, options],
		states_list: [
			Object.entries(stores).map(([plot, value]) => ({id: {type: "plot-req", plot}, value})),
			badges,
		],
	};
	sets.length = 0;
	deh.plotRequest(context, controls, tab, options, hover);
	return Object.fromEntries(sets.map(([id, props]) => [id.plot, props.data]));
};
const uses = ["window", "granularity"];
const drawn = {
	tab: "activity",
	uses,
	context,
	controls: {window: [1, 2], granularity: "day"},
	opts: {metric: "time"},
};
eq(
	request("activity", {
		bar: {tab: "activity", uses},
		line: {tab: "social", uses},
		pie: {tab: "activity", uses},
	}),
	{bar: drawn, pie: Object.assign({}, drawn, {opts: {}})},
	"only the showing tab's cards ask, each with its own options and only the controls it takes"
);
eq(request("activity", {bar: drawn}), {}, "the inputs a card was drawn with ask for nothing");
eq(
	request("activity", {bar: drawn}, metric("time"), {...full, hours: [2, 5], phases: []}),
	{},
	"a control the card ignores asks for nothing"
);
eq(
	request("activity", {bar: drawn}, metric("visits")).bar.opts,
	{metric: "visits"},
	"a changed card option asks again"
);
const social = {tab: "social", uses};
const hovered = {tab: "social", at: 1};
eq(
	request("activity", {bar: drawn, line: social}, metric("time"), full, [], "rec-hover", hovered),
	{line: Object.assign({}, drawn, {tab: "social", opts: {}})},
	"a hovered tab's cards ask for the showing tab's inputs"
);
eq(
	request(
		"activity",
		{bar: drawn, line: social},
		metric("time"),
		{...full, window: [2, 2]},
		[],
		"rec-controls",
		hovered
	),
	{bar: Object.assign({}, drawn, {controls: {window: [2, 2], granularity: "day"}})},
	"a control changed after the hover rebuilds only the showing tab"
);

// --- plotRequest badges: shown while a control the card ignores is narrowed -----------
const badge = (control, value) => ({id: {type: "badge", plot: "bar", control}, value});
const shown = (controls, badges) => {
	request("activity", {}, [], controls, badges);
	return Object.fromEntries(sets.map(([id, props]) => [id.control, props.hidden]));
};
const all = ["window", "hours", "phases", "group_mean"].map((control) => badge(control, true));
eq(shown(full, all), {}, "every control at full extent leaves the badges hidden");
eq(
	shown({...full, window: [2, 2], hours: [0, 11], phases: ["dark_phase"], group_mean: true}, all),
	{window: false, hours: false, phases: false, group_mean: false},
	"each narrowed control shows its badge"
);
eq(
	shown(Object.assign({}, full, {granularity: "phase_count", window: [1, 4]}), all),
	{},
	"a window over every phase is the whole recording"
);
eq(
	shown(full, all.map((b) => Object.assign({}, b, {value: false}))),
	{window: true, hours: true, phases: true, group_mean: true},
	"a widened control hides its badge again"
);
sets.length = 0;
deh.plotRequest(null, null, "activity");
eq(sets.length, 0, "no recording, no request");

// --- restyleTheme: drawn figures repainted in place, the showing tab only ---------------
const looks = {
	dark: {
		template: {layout: {font: {color: "#eee"}}},
		bands: {dark_phase: "#d1", light_phase: "#d2"},
		ink2: "#dink",
	},
};
const lightFig = {
	data: [],
	layout: {
		template: {layout: {font: {color: "#111"}}},
		shapes: [
			{name: "phase-band-dark_phase", fillcolor: "#l1"},
			{name: "event-span", line: {color: "#link", width: 1}},
			{name: "other", fillcolor: "#keep"},
		],
	},
};
const restyle = (figures, tab = "activity") => {
	sets.length = 0;
	const ids = figures.map((_, i) => ({type: "plot", plot: "p" + i}));
	deh.restyleTheme("dark", tab, figures, ids, ids.map(() => ({tab: "activity"})), ids, looks);
	return sets.map(([, props]) => props.figure);
};
const [repainted] = restyle([lightFig, null]);
eq(repainted.layout.template, looks.dark.template, "the template is swapped");
eq(
	repainted.layout.shapes,
	[
		{name: "phase-band-dark_phase", fillcolor: "#d1"},
		{name: "event-span", line: {color: "#dink", width: 1}},
		{name: "other", fillcolor: "#keep"},
	],
	"phase bands and event outlines take the theme's colours, other shapes are kept"
);
eq(restyle([repainted]).length, 0, "a figure already in the theme is left alone");
eq(restyle([lightFig], "social").length, 0, "a hidden tab's figure waits until it shows");

// --- colorBy: the cohort cards redraw for a new colour only ---------------------------
eq(deh.colorBy({color_by: "sex"}, "animal_id"), "sex", "colorBy passes a new colour on");
eq(
	deh.colorBy({color_by: "sex", hours: [2, 5]}, "sex") === NO,
	true,
	"colorBy ignores the other controls"
);

// --- timelineZoom: only a change of the x range asks for the timeline again -----------
const span = ["2023-05-24 12:00", "2023-05-24 13:00"];
const zoom = {"xaxis.range[0]": span[0], "xaxis.range[1]": span[1]};
eq(deh.timelineZoom(zoom, null), span, "a zoom or pan asks for its span");
eq(deh.timelineZoom({"xaxis.range": span}, null), span, "so does a range set whole");
eq(deh.timelineZoom(zoom, span) === NO, true, "the span already drawn asks for nothing");
const reset = {"xaxis.autorange": true, "yaxis.autorange": true};
eq(deh.timelineZoom(reset, span), null, "a reset asks for the window");
eq(deh.timelineZoom(reset, undefined) === NO, true, "the window already drawn asks for nothing");
eq(deh.timelineZoom({autosize: true}, span) === NO, true, "a resize asks for nothing");

// --- clickEvent: a re-rendered button is not a click ----------------------------------
const chip = {type: "chip-x", shelf: "y", field: "day"};
fire(chip, null);
eq(deh.clickEvent() === NO, true, "a re-render passes nothing on");
fire(chip, 1);
eq(deh.clickEvent().id, chip, "a click passes its id on");
eq(deh.clickEvent() === NO, true, "the same click again at once is a double click");
fire({...chip, field: "hour"}, 1);
eq(deh.clickEvent().id.field, "hour", "a different button still passes");

// --- builderKind: the palette redraws for a new plot type only ------------------------
eq(deh.builderKind({kind: "box"}, "line"), "box", "builderKind passes a new type on");
eq(
	deh.builderKind({kind: "box", channels: {}}, "box") === NO,
	true,
	"builderKind ignores other edits"
);

// --- pageScroll: an opened recording takes the builder link to its project -------------
global.history = {};
global.requestAnimationFrame = () => {};
document.getElementById = () => null;
const pages = ["/recording", "/builder"].map((index) => ({id: {index}}));
window.dash_clientside.callback_context = {outputs_list: [pages, pages]};
const links = (pathname, search, hrefs) => deh.pageScroll(pathname, search, hrefs)[1];
eq(
	links("/recording", "?project=b&recording=r", ["/recording", "/builder?project=a"]),
	["/recording?project=b&recording=r", "/builder?project=b"],
	"a recording from another project moves the builder"
);
eq(
	links("/builder", "?project=c", ["/recording?project=b&recording=r", "/builder?project=b"]),
	["/recording?project=b&recording=r", "/builder?project=c"],
	"the builder keeps its own pick"
);
eq(
	links("/recording", "?project=b&recording=s", ["/recording", "/builder?project=c"]),
	["/recording?project=b&recording=s", "/builder?project=c"],
	"another recording of the same project leaves the builder alone"
);

console.log("ok");
"""
)


def _run(tmp_path: Path, harness: str, wanted: dict) -> None:
	script, data = tmp_path / "harness.js", tmp_path / "wanted.json"
	script.write_text(harness, encoding="utf-8")
	data.write_text(json.dumps(wanted), encoding="utf-8")
	result = subprocess.run(
		["node", str(script), str(CLIENTSIDE_JS), str(data)],
		capture_output=True,
		text=True,
		encoding="utf-8",
		check=False,
	)
	assert result.returncode == 0, result.stderr
	assert result.stdout.strip() == "ok"


def test_recording_clientside_callbacks(tmp_path):
	wanted = {
		"days": 5,
		"phases": 13,
		"filename": "cohort 1 wt__activity-bar".replace(" ", "-"),
	}
	_run(tmp_path, _HARNESS, wanted)


_FORMAT_HARNESS = (
	_PRELUDE
	+ """
const NO_UPDATE = window.dash_clientside.no_update;
const sets = [];
window.dash_clientside.set_props = (id, props) => sets.push([id, props]);
const ids = [{plot: "p"}];
const apply = (fig, fmt, choices = wanted.colors) => {
	sets.length = 0;
	deh.applyFormat({p: fmt}, [fig], ids, choices);
	return sets.length ? sets[0][1].figure : null;
};
const titles = (layout) => Object.fromEntries(
	Object.keys(layout).filter((key) => /^[xy]axis\\d*$/.test(key))
		.map((key) => [key, (layout[key].title || {}).text || ""])
);
const bar = ({coloraxis: c}) => [c.cauto, c.cmin, c.cmax, c.colorbar.title.text, c.colorscale];
// Flattened rather than compared whole: plotly.py writes autorangeoptions' two keys the
// other way round, and an unset bound it leaves out entirely.
const ranges = (layout) => Object.fromEntries(
	Object.keys(layout).filter((key) => /^[xy]axis\\d*$/.test(key)).map((key) => {
		const axis = layout[key], opts = axis.autorangeoptions || {};
		const bounds = [opts.minallowed ?? null, opts.maxallowed ?? null];
		return [key, [axis.autorange ?? null, axis.range ?? null, ...bounds]];
	})
);
const edit = (key, value, store, fig = wanted.figure, choices = wanted.colors) => {
	window.dash_clientside.callback_context = {
		triggered_id: {type: "rec-fmt", key},
		triggered: [{prop_id: JSON.stringify({key, type: "rec-fmt"}) + ".value", value}],
	};
	return deh.editFormat(null, null, "p", store, [fig], ids, choices);
};

// --- applyFormat draws each override, and a facet's titles where plotly express put them --
const formatted = apply(wanted.figure, wanted.fmt);
eq(
	titles(formatted.layout),
	{xaxis: "X", yaxis: "Y", xaxis2: "X", yaxis2: ""},
	"titles land on the titled axes only"
);
eq(
	bar(formatted.layout),
	[false, 1, 3, "N", wanted.colors.colorscale.Viridis],
	"the colour axis takes its title, bounds and scale"
);
eq(
	ranges(formatted.layout),
	{
		xaxis: [true, null, 1, 4],
		yaxis: [true, null, 0.5, null],
		xaxis2: [true, null, 1, 4],
		yaxis2: [true, null, 0.5, null],
	},
	"bounds hold on every facet, one alone leaving the data the other"
);
eq(apply(formatted, wanted.fmt), null, "re-applying the same format sets nothing");

const cleared = apply(formatted, {});
eq(titles(cleared.layout), titles(wanted.figure.layout), "clearing restores the server's titles");
eq(bar(cleared.layout), bar(wanted.figure.layout), "clearing restores the server's colour axis");
eq(ranges(cleared.layout), ranges(wanted.figure.layout), "clearing restores the server's ranges");

// a plot that drew its own range gives it up while a bound holds, and takes it back after
const drawn = {data: [], layout: {xaxis: {title: {text: "Hour"}, range: [-0.5, 23.5]}}};
const bounded = apply(drawn, {xmin: {on: "Hour", value: 2}});
eq(
	ranges(bounded.layout).xaxis,
	[true, null, 2, null],
	"a bound hands the server's range back to autorange"
);
eq(apply(bounded, {}).layout.xaxis.range, [-0.5, 23.5], "clearing gives the server's range back");

eq(
	apply(wanted.figure, {xaxis: {on: "another title", value: "X"}}),
	null,
	"a lapsed override draws nothing"
);
eq(apply(wanted.figure, {}), null, "an unformatted figure is left alone");

const on = wanted.auto.colorbar;
const inverted = apply(wanted.figure, {cmin: {on, value: 5}, cmax: {on, value: 2}});
eq(bar(inverted.layout), bar(wanted.figure.layout), "an inverted range is not drawn");

const scaled = apply(wanted.figure, {colorscale: {on: "", value: "Viridis"}});
eq(
	scaled.layout.coloraxis.colorscale,
	wanted.colors.colorscale.Viridis,
	"the colour scale is swapped"
);

// the recording plots bold their titles with <b>, and an override keeps that weight
const bold = {
	data: [],
	layout: {xaxis: {title: {text: "<b>Time</b>"}}, yaxis: {title: {text: "<b>Rank</b>"}}},
};
eq(
	apply(bold, {yaxis: {on: "<b>Rank</b>", value: "Score"}}).layout.yaxis.title.text,
	"<b>Score</b>",
	"an override keeps the bold"
);

// --- editFormat binds each edit to the automatic text it replaced ---------------------
const store = edit("yaxis", "  Visits  ", null);
eq(
	store,
	{p: {yaxis: {on: wanted.auto.yaxis, value: "Visits"}}},
	"an edit binds to the automatic title"
);
eq(edit("yaxis", wanted.auto.yaxis, store), {}, "typing the automatic title back clears it");
eq(edit("yaxis", "Visits", store) === NO_UPDATE, true, "an unchanged value is no update");

const lapsed = {p: {xaxis: {on: "another title", value: "Old"}}};
eq(edit("xaxis", "", lapsed) === NO_UPDATE, true, "an empty field keeps a lapsed override");

const flagged = (key) => sets.find(([id]) => id.key === key)[1].error;
sets.length = 0;
edit("cmax", 0.5, {p: {cmin: {on, value: 1}}});
eq(flagged("cmax"), "Must be above min", "an inverted range is flagged on the form");
sets.length = 0;
edit("xmax", 0.5, {p: {xmin: {on: wanted.auto.xaxis, value: 1}}});
eq(flagged("xmax"), "Must be above min", "an inverted x range is flagged too");

// --- the builder's plot title, kept in its state with the rest of its Format -----------
const placed = {data: [], layout: {title: {x: 0, xanchor: "left"}}};
const titled = apply(placed, {title: {on: "", value: "T"}});
eq(titled.layout.title, {x: 0, xanchor: "left", text: "T"}, "a title keeps the server's placement");
eq(apply(titled, {}).layout.title, placed.layout.title, "clearing it restores the server's");

const builderEdit = (key, value, state, trigger = {type: "builder-fmt", key}) => {
	window.dash_clientside.callback_context = {
		triggered_id: trigger,
		triggered: [{prop_id: JSON.stringify(trigger) + ".value", value}],
		inputs_list: [[{id: {type: "builder-fmt", key: "title"}}]],
	};
	return deh.editBuilderFormat(null, null, true, state, placed, wanted.colors);
};
const withTitle = builderEdit("title", " T ", {kind: "bar"});
eq(withTitle, {kind: "bar", format: {title: {on: "", value: "T"}}}, "an edit lands in the state");
eq(builderEdit(null, 1, withTitle, "builder-fmt-reset"), {kind: "bar"}, "reset leaves no format");

// the builder's dialog mounts with its page, which fires every field and the reset at once
window.dash_clientside.callback_context = {
	triggered_id: {type: "builder-fmt", key: "title"},
	triggered: [
		{prop_id: JSON.stringify({key: "title", type: "builder-fmt"}) + ".value", value: ""},
		{prop_id: "builder-fmt-reset.n_clicks", value: null},
	],
};
prevents(
	() => deh.editBuilderFormat(null, null, false, withTitle, placed, wanted.colors),
	"mounting the closed dialog keeps every override"
);

// --- a palette swaps the colours the colorway declares, and only those -----------------
const A = "rgb(10, 20, 30)", B = "rgb(40, 50, 60)", EDGE = "rgb(1, 2, 3)";
const cats = {
	data: [
		{line: {color: A}, fillcolor: "rgba(10, 20, 30, 0.2)"},
		{marker: {color: [B, A]}, x: [1, 2]},
		{line: {color: EDGE}},
	],
	layout: {colorway: [A, B]},
};
const RED = "rgb(200, 0, 0)", GREEN = "rgb(0, 200, 0)";
const palettes = {palette: {Three: [RED, GREEN, "rgb(0, 0, 200)"], One: ["rgb(9, 9, 9)"]}};
const pick = (name) => ({palette: {on: "", value: name}});

const recoloured = apply(cats, pick("Three"), palettes);
eq(recoloured.data[0].line.color, RED, "a category colour is swapped by its place");
eq(recoloured.data[0].fillcolor, "rgba(200,0,0,0.2)", "its shade keeps the alpha");
eq(recoloured.data[1].marker.color, [GREEN, RED], "per-point colours are swapped too");
eq(recoloured.data[2].line.color, EDGE, "a colour outside the colorway is left alone");
eq(recoloured.layout.colorway, [RED, GREEN], "the colorway follows");
eq(apply(recoloured, pick("Three"), palettes), null, "re-applying the palette sets nothing");

const restored = apply(recoloured, {}, palettes);
eq(
	[restored.data[0].line.color, restored.data[1].marker.color, restored.layout.colorway],
	[A, [B, A], [A, B]],
	"clearing restores the server's colours"
);

eq(
	apply(cats, pick("One"), palettes).layout.colorway,
	[A, B],
	"a palette too short for the categories is not drawn"
);
sets.length = 0;
edit("palette", "One", null, cats, palettes);
eq(flagged("palette"), "1 colours for 2 categories", "and the form says why");

window.dash_clientside.callback_context = {
	triggered_id: {type: "card-format", plot: "p"},
	triggered: [{value: 1}],
	outputs_list: [null, null, null, [{id: {type: "rec-fmt", key: "palette"}}]],
};
eq(
	deh.openFormat(null, [cats], ids, ["P"], ids, {}, palettes)[6],
	[
		{value: "Three", label: "Three · 3", disabled: false},
		{value: "One", label: "One · 1", disabled: true},
	],
	"the palette menu disables the ones too short for this plot"
);

// --- a colour scale resamples the network's edges, whose lines cannot ride the axis ----
const channels = (color) => color.match(/[\\d.]+/g).map(Number);
const near = (got, want) => got.every((color, i) =>
	channels(color).every((c, j) => Math.abs(c - channels(want[i])[j]) <= 1)
);
const edgeLines = (fig) => fig.data.slice(0, -1).map((trace) => trace.line.color);
const viridis = apply(wanted.network, {colorscale: {on: "", value: "Viridis"}});
eq(near(edgeLines(viridis), wanted.networkViridis), true, "edge lines follow the picked scale");
eq(viridis.data.at(-1), wanted.network.data.at(-1), "the nodes are left alone");
eq(
	near(edgeLines(apply(viridis, {})), edgeLines(wanted.network)),
	true,
	"clearing restores the server's edge colours"
);

const fields = ["colorbar", "cmin", "colorscale"].map((key) => ({id: {type: "rec-fmt", key}}));
window.dash_clientside.callback_context.outputs_list = [null, null, null, fields];
eq(
	deh.openFormat(null, [wanted.network], ids, ["P"], ids, {}, wanted.colors)[5],
	[{display: "none"}, {display: "none"}, null],
	"a hidden colour bar offers its scale but no title or bounds"
);

// --- shared y links or frees the facets' y axes ------------------------------------------
const shareY = (value) => ({sharey: {on: "True", value}});
const freed = apply(wanted.figure, shareY("False"));
const freedY = freed.layout.yaxis2;
eq([freedY.matches, freedY.showticklabels], [undefined, true], "False frees the facets");
const relinked = apply(freed, shareY("True")).layout.yaxis2;
eq([relinked.matches, relinked.showticklabels], ["y", false], "True links them back");
const unshared = apply(freed, {}).layout.yaxis2;
eq([unshared.matches, unshared.showticklabels], ["y", false], "clearing restores the drawn axis");
eq(apply(wanted.network, shareY("False")), null, "a single y axis offers no sharing");

console.log("ok");
"""
)


_PATCH_HARNESS = """
global.window = {};
const {readFileSync} = require("fs");
const fakeStore = (hashes) => {
	const seen = [];
	const dispatch = (action) => (seen.push(action.type), action);
	return {seen, getState: () => ({layoutHashes: hashes}), dispatch};
};
const early = fakeStore({"0,props,children,1": {hash: 1}});
window.dash_stores = [early];
eval(readFileSync(process.argv[2], "utf8"));
const late = fakeStore({});
window.dash_stores.push(late);

const reset = (path) => ({type: "RESET_COMPONENT_STATE", payload: {itempath: path}});
early.dispatch(reset(["0", "props", "children", 2]));
early.dispatch(reset(["0", "props", "children"]));
early.dispatch({type: "ON_PROP_CHANGE", payload: {}});
late.dispatch(reset(["0"]));
const want = JSON.stringify([["RESET_COMPONENT_STATE", "ON_PROP_CHANGE"], []]);
const got = JSON.stringify([early.seen, late.seen]);
if (got !== want) throw new Error(got + " !== " + want);
console.log("ok");
"""


def test_renderer_patch_drops_only_resets_with_nothing_to_reset(tmp_path):
	patch = CLIENTSIDE_JS.with_name("renderer_patch.js")
	script = tmp_path / "patch.js"
	script.write_text(_PATCH_HARNESS, encoding="utf-8")
	result = subprocess.run(
		["node", str(script), str(patch)],
		capture_output=True,
		text=True,
		encoding="utf-8",
		check=False,
	)
	assert result.returncode == 0, result.stderr
	assert result.stdout.strip() == "ok"


def test_format_draws_and_clears(tmp_path):
	import plotly.express as px
	import polars as pl

	from deepecohab.plotting import plot_factory, prepare
	from deepecohab.plotting.theme import COLORSCALES, PALETTES

	frame = pl.DataFrame({"a": [1, 2, 3, 4], "b": [1, 2, 1, 2], "g": ["p", "q", "p", "q"]})
	# Faceted, so the x title sits on several axes and the y title on one.
	fig = px.density_heatmap(frame, x="a", y="b", facet_col="g")
	# Each override is set on the text plotly express gave its element.
	auto = {"xaxis": "a", "yaxis": "b", "colorbar": "count"}
	fmt = {
		"xaxis": {"on": "a", "value": "X"},
		"yaxis": {"on": "b", "value": "Y"},
		"colorbar": {"on": "count", "value": "N"},
		"xmin": {"on": "a", "value": 1},
		"xmax": {"on": "a", "value": 4},
		"ymin": {"on": "b", "value": 0.5},
		"cmin": {"on": "count", "value": 1},
		"cmax": {"on": "count", "value": 3},
		"colorscale": {"on": "", "value": "Viridis"},
	}

	nodes, edges = prepare._network_layout(
		pl.DataFrame({"source": ["a", "b"], "target": ["b", "c"], "chasings": [3.0, 1.0]}),
		"chasings",
		True,
		{"a": 30.0, "b": 20.0, "c": 10.0},
		"circular",
		0,
	)
	network = plot_factory.plot_network_graph(
		nodes,
		edges,
		["#111111", "#222222", "#333333"],
		["#111111", "#222222", "#333333"],
		["a", "b", "c"],
		"<b>Network</b>",
		include_ranking=True,
	)

	wanted = {
		"figure": json.loads(fig.to_json()),
		"auto": auto,
		"fmt": fmt,
		"colors": {"colorscale": COLORSCALES, "palette": PALETTES},
		"network": json.loads(to_json_plotly(network)),
		"networkViridis": px.colors.sample_colorscale(
			COLORSCALES["Viridis"], [edge["marker"]["color"][0] for edge in network["data"][:-1]]
		),
	}
	_run(tmp_path, _FORMAT_HARNESS, wanted)


def test_dnd_palette_matches_python():
	from deepecohab.app.pages.builder import PALETTE

	dnd = CLIENTSIDE_JS.with_name("dnd.js").read_text(encoding="utf-8")
	assert re.search(r'var PALETTE = "([^"]+)"', dnd).group(1) == PALETTE
