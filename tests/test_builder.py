"""Tests for the plot builder's field catalog and figure aggregation.

A small synthetic project table stands in for a real one, shaped like
``Project.load_project_table()``'s long format: one row per animal, hour and metric.
"""

import polars as pl
import pytest

from deepecohab.app.builder import catalog, figure


@pytest.fixture
def frame() -> pl.LazyFrame:
	return pl.DataFrame(
		{
			"value": [1.0, 3.0, 2.0, 2.0],
			"exposure": [1.0, 1.0, 1.0, 1.0],
			"metric": ["activity", "activity", "time_alone", "time_alone"],
			"day": [1, 1, 1, 1],
			"hour": [0, 1, 0, 1],
			"phase_count": [1, 1, 1, 1],
			"phase": ["light_phase", "light_phase", "light_phase", "light_phase"],
			"recording": ["rec1", "rec1", "rec1", "rec1"],
			"animal_id": ["A", "B", "A", "B"],
			"genotype": ["WT", "HET", "WT", "HET"],
			"sex": ["F", "M", "F", "M"],
			"n_mice": [2, 2, 2, 2],
		}
	).lazy()


@pytest.fixture
def fields(frame) -> list[catalog.Field]:
	return catalog.prepare(frame)[1]


def test_prepare_classifies_fields(fields):
	by_name = {field.name: field for field in fields}

	assert by_name["value"].kind == "measure"
	assert by_name["value"].agg == "metric"
	assert by_name["metric"].kind == "dimension"
	assert by_name["metric"].group == "Measure"
	assert by_name["hour"].kind == "time"
	assert by_name["genotype"].kind == "dimension"
	assert by_name["genotype"].group == "Animal"
	assert by_name["recording"].group == "Recording"


def test_build_frame_computes_rate(frame, fields):
	state = {
		"kind": "line",
		"measure_as": "rate",
		"channels": {"x": ["hour"], "y": ["value"]},
		"filters": {"metric": ["activity"]},
	}
	data = figure.build_frame(frame, state, fields)

	assert data.sort("hour")["value"].to_list() == pytest.approx([1.0, 3.0])


@pytest.mark.parametrize(
	("unit", "expected"), [("days", [13, 70]), ("weeks", [2, 10]), ("months", [0, 2])]
)
def test_build_frame_rounds_age(frame, unit, expected):
	aged = frame.with_columns(
		pl.when(pl.col("animal_id") == "A")
		.then(pl.duration(days=13))
		.otherwise(pl.duration(days=70))
		.alias("age")
	)
	aged, fields = catalog.prepare(aged)
	state = {
		"kind": "bar",
		"measure_as": "rate",
		"age_unit": unit,
		"channels": {"x": [catalog.AGE], "y": ["value"]},
		"filters": {"metric": ["activity"]},
	}

	assert (
		figure.build_frame(aged, state, fields).sort(catalog.AGE)[catalog.AGE].to_list() == expected
	)
	assert catalog.in_age_unit(fields, unit)[-1].label == f"Age ({unit})"


def test_warnings_for_blocks_unfaceted_multi_metric(frame, fields):
	state = {
		"kind": "line",
		"measure_as": "rate",
		"channels": {"x": ["hour"], "y": ["value"]},
		"filters": {},
	}
	notes = figure.warnings_for(frame, state, fields)

	assert any(note.blocking for note in notes)


def test_warnings_for_allows_faceted_multi_metric(frame, fields):
	state = {
		"kind": "line",
		"measure_as": "rate",
		"channels": {"x": ["hour"], "y": ["value"], "facet_row": ["metric"]},
		"filters": {},
	}
	notes = figure.warnings_for(frame, state, fields)

	assert not any(note.blocking for note in notes)


def test_build_figure_colours_by_sample_palette(frame, fields):
	state = {
		"kind": "line",
		"measure_as": "rate",
		"channels": {"x": ["hour"], "y": ["value"], "color": ["genotype"]},
		"filters": {"metric": ["activity"]},
	}
	fig, notes = figure.build_figure(frame, state, fields)

	assert notes == []
	colors = {trace.line.color for trace in fig.data}
	assert colors == set(figure.theme.sample_palette(2))


def test_category_axis_survives_an_all_null_first_trace(frame, fields):
	# Genotype HET sorts first and never had the event, so its trace's x is all null.
	evented = frame.with_columns(
		pl.when(pl.col("genotype") == "WT").then(pl.lit("During")).alias("Tone")
	)
	evented, fields = catalog.prepare(evented, ["Tone"])
	state = {
		"kind": "box",
		"measure_as": "rate",
		"channels": {"x": ["Tone"], "y": ["value"], "color": ["genotype"]},
		"filters": {"metric": ["activity"]},
	}
	fig, _ = figure.build_figure(evented, state, fields)

	assert fig.data[0].x.tolist() == [None]
	assert fig.layout.xaxis.type == "category"
	assert fig.layout.yaxis.type is None


def test_heatmap_draws_one_cell_per_value(frame, fields):
	# rec1, first on Y, has no hour 2, and hours run to 11: numeric bins would merge
	# neighbours and text would sort 10 before 2.
	later = frame.with_columns(pl.lit("rec2").alias("recording"), pl.col("hour") + 2)
	tens = frame.with_columns(pl.col("hour") + 10, pl.lit("rec10").alias("recording"))
	data = pl.concat([frame, later, tens])
	state = {
		"kind": "heatmap",
		"measure_as": "rate",
		"channels": {"x": ["hour"], "y": ["recording"], "z": ["value"]},
		"filters": {"metric": ["activity"]},
	}
	fig, _ = figure.build_figure(data, state, fields)

	# A whole-number axis stays numeric, so Format's bounds read as hours.
	assert fig.layout.xaxis.type is None
	assert fig.data[0].xbins.to_plotly_json() == {"start": -0.5, "end": 11.5, "size": 1}
	assert fig.layout.yaxis.type == "category"
	# Plotly express lists Y categories bottom up, so the first sorted one sits on top.
	assert list(fig.layout.yaxis.categoryarray) == ["rec2", "rec10", "rec1"]


@pytest.mark.parametrize("kind", ["box", "violin"])
@pytest.mark.parametrize(("shelf", "color"), [("genotype", ["sex"]), ("value", [])])
def test_point_colour_redraws_every_point_once_in_its_box(frame, fields, kind, shelf, color):
	state = {
		"kind": kind,
		"measure_as": "rate",
		"channels": {
			"y": ["value"],
			"color": color,
			figure.POINT_COLOR: [shelf],
			figure.DETAIL: ["animal_id", "hour"],
		},
		"filters": {"metric": ["activity"]},
	}
	fig, _ = figure.build_figure(frame, state, fields)

	shapes = [trace for trace in fig.data if trace.hoveron != "points" and trace.type == kind]
	points = [trace for trace in fig.data if trace.hoveron == "points"]
	# A box keeps its points, hidden, so its whiskers still stop at the 1.5 IQR fences.
	hidden = [
		trace.marker.opacity == 0 and trace.boxpoints == "all"
		if kind == "box"
		else not trace.points
		for trace in shapes
	]
	assert shapes
	assert all(hidden)
	for shape in shapes:
		own = [trace for trace in points if trace.offsetgroup == shape.offsetgroup]
		assert sum(len(trace.y) for trace in own) == len(shape.y)
	if shelf == "genotype":
		assert {trace.marker.color for trace in points} == set(fig.layout.colorway[-2:])
	else:
		assert fig.layout.coloraxis.cmin == pytest.approx(1.0)


@pytest.mark.parametrize("kind", ["box", "violin"])
def test_exclude_outliers_drops_only_points_past_their_own_box_fences(kind):
	values = [1.0, 2.0, 3.0, 4.0, 100.0, 1.0, 2.0, 3.0, 4.0, 5.0]
	wide = pl.DataFrame(
		{
			"value": values,
			"exposure": [1.0] * 10,
			"metric": ["activity"] * 10,
			"recording": ["rec1"] * 10,
			"animal_id": [str(i) for i in range(10)],
			"genotype": ["WT"] * 5 + ["HET"] * 5,
		}
	).lazy()
	wide, fields = catalog.prepare(wide)
	state = {
		"kind": kind,
		"measure_as": "rate",
		"channels": {"x": ["genotype"], "y": ["value"], figure.DETAIL: ["recording", "animal_id"]},
		"filters": {"metric": ["activity"]},
	}

	kept = figure.build_frame(wide, state, fields).remove(figure.outliers(state, fields))
	assert sorted(kept.filter(pl.col("genotype") == "WT")["value"]) == [1.0, 2.0, 3.0, 4.0]
	assert sorted(kept.filter(pl.col("genotype") == "HET")["value"]) == [1.0, 2.0, 3.0, 4.0, 5.0]

	excluding = dict(state, outliers="exclude")
	fig, notes = figure.build_figure(wide, excluding, fields)
	assert sum(len(trace.y) for trace in fig.data) == 9
	assert notes == [
		f"Excluded 1 of 10 points lying more than 1.5 IQR outside their {kind}'s quartiles."
	]

	# The CSV keeps the excluded point, flagged, under the axis titles.
	data = figure.plotted_data(wide, excluding, fields)
	value = figure.label_for(next(item for item in fields if item.name == "value"), "rate")
	assert data.filter("excluded_outlier")[value].to_list() == [100.0]
	assert data.height == 10


def test_plotted_data_is_none_when_the_plot_cannot_be_drawn(frame, fields):
	state = figure.new_state("box")

	assert figure.plotted_data(frame, state, fields) is None


def test_reduce_dispatches_every_trigger(frame, fields, monkeypatch):
	"""``_reduce`` matches each trigger it handles, and prevents the update on anything else.

	The reducer is one ``match`` over the id Dash triggered with, so a mistyped pattern
	silently falls through to ``PreventUpdate`` rather than failing loudly.
	"""
	import dash
	from dash.exceptions import PreventUpdate

	dash.Dash(__name__, use_pages=True, pages_folder="")  # the page module registers itself

	from deepecohab.app import services
	from deepecohab.app.pages import builder as page

	monkeypatch.setattr(services, "builder_frame", lambda location: (frame, fields))

	def reduce(trigger, value=None, event=None, state=None):
		monkeypatch.setattr(dash._callback_context, "context_value", None, raising=False)
		monkeypatch.setattr(
			page,
			"ctx",
			type("Ctx", (), {"triggered_id": trigger, "triggered": [{"value": value}]}),
		)
		return page._reduce(
			event,
			None,
			None,
			None,
			None,
			None,
			None,
			state if state is not None else figure.new_state(),
			{"location": "somewhere"},
			None,
			[],
			None,
		)

	held = figure.new_state()
	held["channels"] = {"x": ["hour"], "y": ["value"]}
	held["filters"] = {"metric": {"mode": "pick", "values": ["activity"]}}

	assert reduce("builder-clear", state=dict(held, channels={"x": ["hour"]}))[0]["channels"] == {}
	assert reduce({"type": "builder-kind", "kind": "bar"})[0]["kind"] == "bar"
	assert reduce({"type": "builder-mode", "mode": "mean"})[0]["measure_as"] == "mean"
	excluding = reduce({"type": "builder-outliers", "kind": "box"}, value=True)[0]
	assert excluding["outliers"] == "exclude"
	assert (
		"outliers"
		not in reduce({"type": "builder-outliers", "kind": "box"}, value=False, state=excluding)[0]
	)
	assert reduce({"type": "builder-kind", "kind": "bar"})[0]["kind"] == "bar"
	bar = figure.new_state("bar")
	stacked = reduce({"type": "builder-option", "option": "barmode"}, value="stack", state=bar)
	assert stacked[0]["barmode"] == "stack"
	with pytest.raises(PreventUpdate):  # a box option on a bar plot is ignored
		reduce({"type": "builder-option", "option": "boxmode"}, value="overlay", state=bar)
	assert (
		reduce({"type": "chip-x", "shelf": "x", "field": "hour"}, state=held)[0]["channels"]["x"]
		== []
	)
	assert (
		"metric"
		not in reduce({"type": "chip-x", "shelf": page.FILTERS, "field": "metric"}, state=held)[0][
			"filters"
		]
	)
	assert reduce({"type": "chip-send", "field": "hour", "from": page.PALETTE, "to": "x"})[0][
		"channels"
	]["x"] == ["hour"]
	assert reduce("dnd-event", event={"field": "hour", "shelf": "x"})[0]["channels"]["x"] == [
		"hour"
	]
	assert (
		reduce({"type": "filter-mode", "field": "hour", "mode": "range"}, state=held)[0]["filters"][
			"hour"
		]["mode"]
		== "range"
	)
	assert reduce({"type": "filter-pick", "field": "metric"}, value=["time_alone"], state=held)[0][
		"filters"
	]["metric"]["values"] == ["time_alone"]
	assert reduce({"type": "filter-range", "field": "hour"}, value=[0, 1], state=held)[0][
		"filters"
	]["hour"] == {"mode": "range", "lo": 0, "hi": 1}
	assert (
		reduce({"type": "chip-bin", "field": "hour"}, value=" 1-3, 4-6 ", state=held)[0]["bins"][
			"hour"
		]
		== "1-3, 4-6"
	)

	# A Blocks spec outlives the field it was typed on unless something clears it, and
	# would then come back to life under the field when it is dropped again.
	binned = dict(held, channels={"facet_col": ["day"]}, bins={"day": "1-3, 4-6"})
	assert reduce({"type": "builder-mode", "mode": "rate"}, state=binned)[0]["bins"] == {
		"day": "1-3, 4-6"
	}
	unshelved = dict(binned, channels={})
	assert "bins" not in reduce({"type": "builder-mode", "mode": "rate"}, state=unshelved)[0]

	aged = dict(held, channels={"x": [catalog.AGE]})
	weeks = reduce({"type": "age-unit", "unit": "weeks"}, state=aged)[0]
	assert weeks["age_unit"] == "weeks"
	assert "age_unit" not in reduce({"type": "age-unit", "unit": "days"}, state=weeks)[0]
	assert (
		"age_unit"
		not in reduce({"type": "builder-mode", "mode": "rate"}, state=dict(weeks, channels={}))[0]
	)

	for unknown in ("no-such-id", {"type": "no-such-type"}, {"type": "chip-x", "shelf": "x"}):
		with pytest.raises(PreventUpdate):
			reduce(unknown)


def _by_position(frame: pl.LazyFrame) -> pl.LazyFrame:
	"""The fixture with every hour spread unevenly over two positions of different types."""
	return pl.concat(
		[
			frame.with_columns(
				pl.col("value", "exposure") * share,
				pl.lit(position).alias("position"),
				pl.lit(kind).alias("position_type"),
			)
			for position, kind, share in (("cage_1", "home", 0.25), ("cage_2", "stimulus", 0.75))
		]
	)


@pytest.mark.parametrize("mode", list(figure.MEASURE_MODES))
def test_an_hour_spread_over_positions_reads_the_same_in_every_mode(frame, fields, mode):
	"""Summing the positions back is what every mode must do, the hourly mean included."""
	state = {
		"kind": "line",
		"measure_as": mode,
		"channels": {"x": ["genotype"], "y": ["value"]},
		"filters": {"metric": ["activity"]},
	}
	split = _by_position(frame)
	whole = figure.build_frame(frame, state, fields).sort("genotype")
	parts = figure.build_frame(split, state, catalog.prepare(split)[1]).sort("genotype")

	assert parts["value"].to_list() == pytest.approx(whole["value"].to_list())


@pytest.mark.parametrize("mode", list(figure.MEASURE_MODES))
def test_hover_lists_fields_without_changing_what_is_drawn(frame, mode):
	split = _by_position(frame)
	fields = catalog.prepare(split)[1]
	state = {
		"kind": "line",
		"measure_as": mode,
		"channels": {"x": ["genotype"], "y": ["value"]},
		"filters": {"metric": ["activity"]},
	}
	bare = figure.build_frame(split, state, fields).sort("genotype")
	state["channels"][figure.HOVER] = ["position", "animal_id", "hour", "n_mice"]
	hovered = figure.build_frame(split, state, fields).sort("genotype")

	assert hovered["value"].to_list() == pytest.approx(bare["value"].to_list())
	assert hovered.select("position", "animal_id", "hour").rows() == [
		("cage_1, cage_2", "B", "1"),
		("cage_1, cage_2", "A", "0"),
	]


def test_position_type_filter_keeps_only_those_positions(frame):
	split = _by_position(frame)
	fields = catalog.prepare(split)[1]
	state = {
		"kind": "line",
		"measure_as": "total",
		"channels": {"x": ["genotype"], "y": ["value"]},
		"filters": {"metric": ["activity"], "position_type": ["stimulus"]},
	}
	data = figure.build_frame(split, state, fields).sort("genotype")

	assert {field.name: field.group for field in fields}["position_type"] == "Position"
	assert dict(data.select("genotype", "value").iter_rows()) == {"HET": 2.25, "WT": 0.75}


def _daily(frame: pl.LazyFrame, days: int) -> pl.LazyFrame:
	"""The fixture repeated over ``days`` days, so a block has something to span."""
	return pl.concat(
		[frame.with_columns(pl.lit(day, dtype=pl.Int64).alias("day")) for day in range(1, days + 1)]
	)


def _binned(frame, fields, spec: str):
	state = {
		"kind": "line",
		"measure_as": "total",
		"channels": {"x": ["day"], "y": ["value"]},
		"filters": {"metric": ["activity"]},
		"bins": {"day": spec},
	}
	return figure.build_frame(frame, state, fields), state


@pytest.mark.parametrize(
	("spec", "expected"),
	[
		("3", ["day 1-3", "day 4-6"]),
		("5", ["day 1-5", "day 6-10"]),
		("1-3, 4-6", ["day 1-3", "day 4-6"]),
		# Uneven spans, more than two of them, in the order typed rather than sorted.
		("4-6, 1-2, 3-3", ["day 4-6", "day 1-2", "day 3-3"]),
	],
)
def test_build_frame_bins_ordered_field(frame, fields, spec, expected):
	data, _state = _binned(_daily(frame, 6), fields, spec)

	assert data["day"].to_list() == expected


def test_build_frame_drops_values_outside_every_span(frame, fields):
	data, _state = _binned(_daily(frame, 6), fields, "1-2, 5-6")

	assert data["day"].to_list() == ["day 1-2", "day 5-6"]
	# 4 rows/day over 2 days, activity only: (1.0 + 3.0) per day.
	assert data["value"].to_list() == pytest.approx([8.0, 8.0])


def test_unreadable_block_spec_warns_without_blocking(frame, fields):
	_data, state = _binned(_daily(frame, 6), fields, "every other")
	notes = figure.warnings_for(frame, state, fields)

	assert any("could not be read" in note.text and not note.blocking for note in notes)


@pytest.mark.parametrize(
	("spec", "expected"),
	[("3", 3), ("1", None), ("", None), ("2-4", [(2.0, 4.0)]), ("4-2", None), ("x-y", None)],
)
def test_parse_bins(spec, expected):
	assert figure.parse_bins(spec) == expected


def test_single_metric_chips_put_two_metrics_on_one_plot(frame, fields):
	"""Each chip reads only its own metric's rows, whatever the Metric filter holds."""
	state = {
		"kind": "scatter",
		"measure_as": "rate",
		"channels": {
			"x": ["metric:activity"],
			"y": ["metric:time_alone"],
			figure.DETAIL: ["animal_id"],
		},
		"filters": {"metric": ["time_alone"]},
	}
	data = figure.build_frame(frame, state, fields)

	assert data["metric:activity"].to_list() == pytest.approx([1.0, 3.0])
	assert data["metric:time_alone"].to_list() == pytest.approx([2.0, 2.0])
	assert figure.warnings_for(frame, state, fields) == []


@pytest.mark.parametrize("pooled", [{"y": ["value"]}, {"facet_row": ["metric"]}])
def test_single_metric_chip_blocks_beside_value_or_metric(frame, fields, pooled):
	state = {
		"kind": "scatter",
		"measure_as": "rate",
		"channels": {"x": ["metric:activity"], **pooled},
		"filters": {},
	}

	assert any(note.blocking for note in figure.warnings_for(frame, state, fields))


def test_heatmap_draws_one_cell_per_hour_averaging_detail(frame, fields):
	state = {
		"kind": "heatmap",
		"measure_as": "rate",
		"channels": {"x": ["hour"], "y": ["genotype"], "z": ["value"]},
		"filters": {"metric": ["activity"]},
	}
	fig, _notes = figure.build_figure(frame, state, fields)

	assert fig.data[0].histfunc == "avg"
	assert list(fig.data[0].x) == [0, 1]
	assert fig.data[0].xbins.size == 1


def test_heatmap_blocks_a_measure_on_an_axis(frame, fields):
	state = {
		"kind": "heatmap",
		"measure_as": "rate",
		"channels": {"x": ["metric:activity"], "y": ["animal_id"], "z": ["metric:time_alone"]},
		"filters": {},
	}

	assert any(note.blocking for note in figure.warnings_for(frame, state, fields))


@pytest.mark.parametrize(
	("kind", "argument", "chosen", "read"),
	[
		("bar", "barmode", "stack", lambda fig: fig.layout.barmode),
		("box", "boxmode", "overlay", lambda fig: fig.layout.boxmode),
		("line", "line_shape", "spline", lambda fig: fig.data[0].line.shape),
	],
)
def test_layout_option_reaches_the_figure(frame, fields, kind, argument, chosen, read):
	state = figure.new_state(kind)
	state["channels"] = {"x": ["hour"], "y": ["value"]}
	state["filters"] = {"metric": ["activity"]}
	default, _ = figure.build_figure(frame, state, fields)
	picked, _ = figure.build_figure(frame, dict(state, **{argument: chosen}), fields)
	assert read(picked) == chosen != read(default)
