"""Tests for the clientside callbacks that keep the Projects table's selection.

The pair is a feedback loop — clicks write the store, the store paints the checkboxes, and
each repaint fires the first half again — so the checks below run both halves through a
stand-in for Dash's dispatch loop in node and assert that every click converges.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

CLIENTSIDE_JS = Path(__file__).parent.parent / "deepecohab" / "app" / "assets" / "clientside.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs node")

_HARNESS = """
global.window = {addEventListener: () => {}};
// The file wires a MutationObserver and a scroll listener on load; node has no DOM.
global.document = {body: {}, querySelectorAll: () => [], addEventListener: () => {}};
global.MutationObserver = class {
	observe() {}
};
eval(require("fs").readFileSync(process.argv[2], "utf8"));
const {selectRecordings: select, paintSelection: paint} = window.dash_clientside.deh;
const NO = {__no_update: true};
const names = (st) => st.selection.map((pair) => pair.join("/")).sort();

function app(projects) {
	const boxes = [], heads = [], removes = [];
	for (const [project, recordings] of Object.entries(projects)) {
		heads.push({id: {type: "select-all", index: project}, value: false});
		removes.push({id: {type: "remove-recordings", index: project}, value: true});
		for (const index of recordings) {
			boxes.push({id: {type: "recording-check", project, index}, value: false});
		}
	}
	return {boxes, heads, removes, selection: [], clear: 0};
}

function run(st, trigger, value) {
	const queue = [[trigger, value]];
	for (let step = 0; queue.length; step++) {
		if (step > 200) throw new Error("the callbacks did not converge");
		const [id, val] = queue.shift();
		global.window = {dash_clientside: {no_update: NO, callback_context: {
			triggered_id: id,
			triggered: [{value: val}],
			inputs_list: [st.boxes, st.heads, {id: "selection-clear", value: st.clear}],
		}}};
		const out = select(null, null, st.clear, st.selection);
		if (out === NO) continue;
		st.selection = out;
		// paintSelection sets the boxes rather than returning them, and only the ones
		// whose value actually changes, so set_props collects exactly the repaints.
		const painted = [];
		global.window = {dash_clientside: {
			set_props: (id, props) => "hidden" in props
				? (st.removes.find((b) => b.id.index === id.index).value = props.hidden)
				: painted.push([id, props.checked]),
			callback_context: {states_list: [st.boxes, st.heads, st.removes]},
		}};
		paint(st.selection);
		painted.forEach(([id, checked]) => {
			const box = [...st.boxes, ...st.heads]
				.find((candidate) => JSON.stringify(candidate.id) === JSON.stringify(id));
			box.value = checked;
			queue.push([box.id, checked]);  // a repaint is an Input too
		});
	}
	return st;
}

function click(st, id, value) {  // the browser ticks the box, then Dash fires
	[...st.boxes, ...st.heads]
		.find((box) => JSON.stringify(box.id) === JSON.stringify(id)).value = value;
	return run(st, id, value);
}

const eq = (got, want, what) => {
	if (JSON.stringify(got) !== JSON.stringify(want)) {
		throw new Error(`${what}: ${JSON.stringify(got)} !== ${JSON.stringify(want)}`);
	}
};
const all = (project) => ({type: "select-all", index: project});
const one = (project, index) => ({type: "recording-check", project, index});

let st = app({p1: ["a", "b", "c"], p2: ["x", "y"]});
click(st, all("p1"), true);
eq(names(st), ["p1/a", "p1/b", "p1/c"], "select all of p1");
eq(st.boxes.map((box) => box.value), [true, true, true, false, false], "p1 ticked");
eq(st.heads.map((head) => head.value), [true, false], "p1's header ticked, p2's not");
eq(st.removes.map((button) => button.value), [false, true], "remove shows for p1 only");

click(st, one("p1", "b"), false);
eq(names(st), ["p1/a", "p1/c"], "untick one recording");
click(st, one("p1", "c"), false);
eq(st.removes.map((button) => button.value), [true, true], "one selected hides remove");
click(st, one("p1", "c"), true);
eq(st.heads.map((head) => head.value), [false, false], "the header unticks with a gap");

click(st, one("p1", "b"), true);
eq(st.heads.map((head) => head.value), [true, false], "the header ticks when the gap closes");

click(st, all("p2"), true);
eq(names(st), ["p1/a", "p1/b", "p1/c", "p2/x", "p2/y"], "p2 keeps p1 selected");

click(st, all("p1"), false);
eq(names(st), ["p2/x", "p2/y"], "deselecting p1 keeps p2");
eq(st.boxes.map((box) => box.value), [false, false, false, true, true], "p1 cleared");

st.clear = 1;
run(st, "selection-clear", 1);
eq(names(st), [], "clear empties the store");
eq(st.boxes.map((box) => box.value), [false, false, false, false, false], "clear unticks all");

st = app({p1: ["a"]});
st.selection = [["filtered-out", "z"]];  // a project the search hid keeps its selection
click(st, all("p1"), true);
eq(names(st), ["filtered-out/z", "p1/a"], "a hidden selection survives a click");

console.log("ok");
"""


def test_selection_callbacks(tmp_path):
	harness = tmp_path / "harness.js"
	harness.write_text(_HARNESS, encoding="utf-8")
	result = subprocess.run(
		["node", str(harness), str(CLIENTSIDE_JS)],
		check=False,
		capture_output=True,
		text=True,
		encoding="utf-8",
	)
	assert result.returncode == 0, result.stderr
	assert result.stdout.strip() == "ok"
