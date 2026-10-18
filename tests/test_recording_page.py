"""Tests for the recording dashboard's navigation callbacks."""

import contextlib
import math

import pytest
import strategies
from dash import Dash
from dash._callback_context import context_value
from dash._utils import AttributeDict
from dash.exceptions import PreventUpdate

from deepecohab.app import components, services
from deepecohab.core.data_model import Bout, Cage, Event, Layout, Tunnel

# The page module calls dash.register_page at import time, which needs an app to register
# against; pages_folder="" keeps this one from re-importing the whole pages package.
Dash(__name__, use_pages=True, pages_folder="")

from deepecohab.app.pages import recording  # noqa: E402

_CONTEXT = {"location": "/project", "recording": "rec_b"}


class _Stub:
	def __init__(self):
		self.recordings = [AttributeDict(name=name) for name in ("rec_a", "rec_b", "rec_c")]


@pytest.fixture(autouse=True)
def _project(monkeypatch):
	monkeypatch.setattr(services, "load_project", lambda _location: _Stub())


@contextlib.contextmanager
def _triggered(*inputs):
	"""Dash's context for a callback fired by ``inputs``; none of them means a fresh mount."""
	context_value.set(AttributeDict(triggered_inputs=list(inputs)))
	try:
		yield
	finally:
		context_value.set({})


def test_rebuilt_header_does_not_navigate_on_its_own():
	# rec-select / rec-prev / rec-next are rebuilt into rec-body on every recording switch,
	# so Dash fires this callback with no trigger behind it. Stepping there walked to the
	# previous recording, which rebuilt the header and fired it again.
	with _triggered(), pytest.raises(PreventUpdate):
		recording._switch_recording("rec_b", None, None, _CONTEXT, "?recording=rec_b", None)


def test_next_click_steps_to_the_following_recording_on_the_showing_tab():
	# url.search still holds the tab the page loaded with; rec-tab holds the one showing.
	with _triggered({"prop_id": "rec-next.n_clicks", "value": 1}):
		search = recording._switch_recording(
			"rec_b", None, 1, _CONTEXT, "?recording=rec_b&tab=overview", "social"
		)
	assert search == ("?recording=rec_c&tab=social", "/recording?recording=rec_c&tab=social")


def test_prev_click_wraps_past_the_first_recording():
	context = {"location": "/project", "recording": "rec_a"}
	with _triggered({"prop_id": "rec-prev.n_clicks", "value": 1}):
		search = recording._switch_recording("rec_a", 1, None, context, "?recording=rec_a", None)
	assert search == ("?recording=rec_c", "/recording?recording=rec_c")


def _layout(cages: list[tuple], tunnels: list[tuple]) -> Layout:
	"""A habitat from ``(cell_id, cage_type, antennas)`` and ``(no, start, end, antennas)``."""
	return Layout(
		cages=[
			{"name": f"cage_{cell[1:]}", "cell_id": cell, "cage_type": kind, "antennas": antennas}
			for cell, kind, antennas in cages
		],
		tunnels=[
			{
				"name": f"tunnel_{no}",
				"tunnel_no": no,
				"start_cell_id": start,
				"end_cell_id": end,
				"antennas": antennas,
			}
			for no, start, end, antennas in tunnels
		],
		antenna_combinations={},
		tunnels_map={},
	)


def _square() -> Layout:
	"""The four-cage habitat every config in the wild uses."""
	return _layout(
		[
			("C1", "social", ["1", "8"]),
			("C2", "standard", ["2", "3"]),
			("C3", "nonsocial", ["4", "5"]),
			("C4", "standard", ["6", "7"]),
		],
		[
			(1, "C1", "C2", ["1", "2"]),
			(2, "C2", "C3", ["3", "4"]),
			(3, "C3", "C4", ["5", "6"]),
			(4, "C1", "C4", ["8", "7"]),
		],
	)


def test_four_cages_sit_on_one_circle():
	# No coordinates are in the config: the cage graph's cycle is what puts them on a ring.
	places = components._habitat_positions(_square())

	assert len(set(places.values())) == 4
	radii = {round(math.hypot(x, y), 6) for x, y in places.values()}
	assert len(radii) == 1
	# First cage declared leads, top left, and the rest run clockwise from it.
	assert places["C1"][0] < 0
	assert places["C1"][1] < 0
	assert places["C2"][0] > 0
	assert places["C2"][1] < 0
	assert places["C3"][0] > 0
	assert places["C3"][1] > 0


def test_a_cage_off_the_ring_is_pushed_out_past_its_neighbour():
	# cage_5 hangs off cage_4 rather than closing a second cycle, so it cannot go on the ring.
	pendant = _square()
	pendant.cages.append(
		Cage(name="cage_5", cell_id="C5", cage_type="social", antennas=["10"]),
	)
	pendant.tunnels.append(
		Tunnel(
			name="tunnel_5",
			tunnel_no=5,
			start_cell_id="C4",
			end_cell_id="C5",
			antennas=["9", "10"],
		),
	)
	places = components._habitat_positions(pendant)

	assert len(set(places.values())) == 5
	ring = [math.hypot(*places[cell]) for cell in ("C1", "C2", "C3", "C4")]
	assert len({round(radius, 6) for radius in ring}) == 1
	assert math.hypot(*places["C5"]) > ring[0]


def test_an_eight_cage_ring_draws_as_an_octagon():
	# Nothing about four cages is written in, so a bigger habitat is still one circle.
	cells = [(f"C{i}", "standard", [str(2 * i - 2 or 16), str(2 * i - 1)]) for i in range(1, 9)]
	tunnels = [(i, f"C{i}", f"C{i % 8 + 1}", [str(2 * i - 1), str(2 * i)]) for i in range(1, 9)]
	places = components._habitat_positions(_layout(cells, tunnels))

	assert len(set(places.values())) == 8
	assert len({round(math.hypot(x, y), 6) for x, y in places.values()}) == 1


def test_the_commonest_cage_type_draws_plain_as_the_blueprint_standard_does():
	# Two standard cages, then nonsocial and social by name: plain, sunken, accent.
	assert components._cage_looks(_square()) == {"standard": 0, "nonsocial": 1, "social": 2}


def test_free_text_cage_types_each_get_a_look_until_eight():
	# Types are whatever the config says; past eight the looks repeat.
	cells = [(f"C{i}", f"kind {i}", [str(2 * i - 2 or 18), str(2 * i - 1)]) for i in range(1, 10)]
	tunnels = [(i, f"C{i}", f"C{i % 9 + 1}", [str(2 * i - 1), str(2 * i)]) for i in range(1, 10)]
	layout = _layout(cells, tunnels)

	looks = components._cage_looks(layout)
	_, legend = components.habitat_map(layout, "Habitat of rec_b")

	assert len(set(looks.values())) == 8
	assert looks["kind 9"] == looks["kind 1"]
	assert [span.children[1] for span in legend.children[:9]] == [f"kind {i}" for i in range(1, 10)]


def test_every_antenna_is_tinted_by_its_own_miss_rate():
	# The bands are the header badge's: under 1% plain, 1-2.5% warn, 2.5% and over bad.
	stats = {
		antenna: {"correct": 1200, "interpolated": 3, "bad": 1, "miss": miss}
		for antenna, miss in {"1": 0.4, "2": 1.0, "3": 2.49, "4": 2.5, "5": 9.9}.items()
	}
	svg = components._habitat_svg(_square(), "Habitat of rec_b", stats)

	assert svg.count('class="deh-hab-ant warn"') == 2
	assert svg.count('class="deh-hab-ant bad"') == 2
	# Antenna 1 is under the band, and 6-8 have no rate at all.
	assert svg.count('class="deh-hab-ant"') == 4
	# The click popover carries the reads, escaped once more for its attribute.
	assert "&lt;span&gt;2.49%&lt;/span&gt;" in svg
	assert "&lt;span&gt;1,200&lt;/span&gt;" in svg


def test_antennas_draw_plain_without_a_quality_table():
	# The card still renders for a recording whose pipeline has never run.
	svg = components._habitat_svg(_square(), "Habitat of rec_b", None)

	assert svg.count('class="deh-hab-ant"') == 8
	assert "Missed" not in svg


def test_layout_names_are_escaped_into_the_markup():
	# The SVG is set as innerHTML and every name in it comes from a hand-edited config.json.
	nasty = _square()
	nasty.cages[0].name = '<script>alert("x")</script>'
	svg = components._habitat_svg(nasty, "Habitat of <b>rec</b>", None)

	assert "<script>" not in svg
	assert "&lt;script&gt;" in svg
	assert 'aria-label="Habitat of &lt;b&gt;rec&lt;/b&gt;"' in svg


def test_the_map_carries_its_svg_for_the_painter():
	# Dash has no SVG components, so deh.paintHabitat reads this attribute.
	body, legend = components.habitat_map(_square(), "Habitat of rec_b", height=500)

	assert body.className == "deh-hab"
	assert body.style == {"height": "500px"}
	assert getattr(body, "data-hab").startswith("<svg viewBox=")
	assert legend.className == "deh-hab-legend"
	# The band keys only mean something once there are rates to band by.
	assert len(legend.children) == 5
	_, banded = components.habitat_map(
		_square(),
		"Habitat of rec_b",
		antenna_stats={"1": {"correct": 1, "interpolated": 0, "bad": 0, "miss": 3.0}},
	)
	assert len(banded.children) == 7


def test_events_card_places_each_bout_on_the_recording_clock():
	"""Day and phase count from the experiment start, as the window slider does.

	The fixture runs 72 h from 24 May 00:00 UTC, light from 00:00 and dark from 12:00.
	"""
	at = strategies.at
	stimulus = Bout(start=at(2023, 5, 25, 13), end=at(2023, 5, 25, 15, 30), position=["cage_1"])
	swap = Bout(start=at(2023, 5, 24, 23), end=at(2023, 5, 25, 1))
	events = [
		Event(name="stimulus", description="", bouts=[stimulus]),
		Event(name="swap", description="", bouts=[swap]),
	]
	_, strip, table, _ = recording._events_card_children(
		strategies.analysis_recording(events=events)
	)

	rows = [
		[cell.children for cell in row.children]
		for body in table.children.children[1:]
		for row in body.children[1:]
	]
	assert rows == [
		["Day 2 · dark", "25 May 13:00", "15:30", "2h 30m", "cage 1"],
		["Day 1 · dark", "24 May 23:00", "25 May 01:00", "2h", "Whole habitat"],
	]
	bar = strip.children[1].children[0]
	assert bar.style["left"] == f"max(3px, {100 * 37 / 72:.3f}%)"
	assert bar.style["right"] == f"max(3px, {100 - 100 * 39.5 / 72:.3f}%)"


def test_a_plot_that_fails_to_build_shows_a_message_instead_of_a_blank_card(monkeypatch):
	def boom(*_args, **_kwargs):
		raise RuntimeError("bad data")

	monkeypatch.setattr(services, "plot_options", boom)

	figure = recording._build_plot("activity-bar", {}, True, "light")

	assert [note["text"] for note in figure["layout"]["annotations"]] == ["Plot creation failed"]


def test_a_plot_prevented_from_updating_stays_prevented(monkeypatch):
	def prevented(*_args, **_kwargs):
		raise PreventUpdate

	monkeypatch.setattr(services, "plot_options", prevented)

	with pytest.raises(PreventUpdate):
		recording._build_plot("activity-bar", {}, True, "light")


def test_a_cage_no_tunnel_reaches_is_not_a_habitat():
	cages = [_cage(n) for n in range(1, 4)]
	chain = [_tunnel(1, "C1", "C2"), _tunnel(2, "C2", "C3")]

	assert _habitat(cages, chain)
	with pytest.raises(ValueError, match="cage_3"):
		_habitat(cages, chain[:1])
	with pytest.raises(ValueError, match="C9"):
		_habitat(cages, [*chain, _tunnel(9, "C1", "C9")])


def _cage(number: int) -> dict:
	return {
		"name": f"cage_{number}",
		"cell_id": f"C{number}",
		"cage_type": "standard",
		"antennas": [],
	}


def _tunnel(number: int, start: str, end: str) -> dict:
	return {
		"name": f"tunnel_{number}",
		"tunnel_no": number,
		"start_cell_id": start,
		"end_cell_id": end,
		"antennas": [],
	}


def _habitat(cages: list[dict], tunnels: list[dict]) -> Layout:
	return Layout(cages=cages, tunnels=tunnels, antenna_combinations={}, tunnels_map={})
