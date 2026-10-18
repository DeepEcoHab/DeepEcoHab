"""Plotting-layer tests that need no recording on disk.

The end-to-end plot tests only check that every plot builds, so the rules the plotting
layer relies on are pinned here: the palette, the duration rendering, the registry's
registration-time guards, and core's independence from plotting.
"""

import base64
import copy
import datetime as dt
import json
import subprocess
import sys
from typing import Literal
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest
import strategies
from plotly.io.json import to_json_plotly

from deepecohab.plotting import (
	animals as animals_module,
	durations,
	plot_factory,
	prepare,
	theme,
)
from deepecohab.plotting.context import PlotContext
from deepecohab.plotting.prepare import Heatmap
from deepecohab.plotting.registry import Figure, PlotRegistry

ANIMALS = ["0035A", "0035B", "0035C", "0035D"]


@pytest.fixture
def context() -> PlotContext:
	"""A context over a four-animal cohort split by sex and treatment."""
	cohort = pl.DataFrame(
		{
			"animal_id": ANIMALS,
			"subject_name": ["m1", "m2", "m3", "m4"],
			"sex": ["M", "M", "F", "F"],
			"genotype": ["WT", "WT", "WT", "WT"],
			"treatment": ["vehicle", "drug", "vehicle", "drug"],
			"mouse_line": ["l"] * 4,
			"genetic_background": ["b"] * 4,
		}
	)

	return PlotContext(
		_loaded={"animals": cohort},
		animal_ids=list(ANIMALS),
		cages=["cage_1", "cage_2"],
		# As layout.positions_non_directional carries them: cages, tunnels, sentinel.
		positions=["cage_1", "cage_2", "tunnel_1", "tunnel_2", "undefined"],
		phases={"light_phase": 0.0, "dark_phase": 12.0},
		days_range=(1, 3),
		phase_range=(1, 6),
		tunnels_map={"tunnel_1_a": "tunnel_1", "tunnel_1_b": "tunnel_1"},
	)


def decoded(array: dict | list) -> np.ndarray:
	"""A figure's array as the browser reads it, a base64 typed array decoded."""
	if not isinstance(array, dict):
		return np.asarray(array)
	values = np.frombuffer(base64.b64decode(array["bdata"]), array["dtype"])
	shape = array.get("shape")
	return values if shape is None else values.reshape([int(n) for n in str(shape).split(",")])


# --- palette -----------------------------------------------------------------


@pytest.mark.parametrize("count", [1, 2, 4, 12, 13, 24])
def test_palette_returns_one_distinct_colour_per_value(count):
	"""Phase is cyclic, so the closing endpoint must not repeat the first colour."""
	colors = theme.sample_palette(count)

	assert len(colors) == count
	assert len(set(colors)) == count


def test_palette_is_empty_for_no_values():
	"""An empty cohort asks for no colours rather than raising."""
	assert theme.sample_palette(0) == []


# --- durations ---------------------------------------------------------------


def _frame(values: list[dt.timedelta]) -> pl.DataFrame:
	return pl.DataFrame({"t": pl.Series(values, dtype=pl.Duration("us"))})


@pytest.mark.parametrize(
	("value", "expected"),
	[
		(dt.timedelta(seconds=4), "seconds"),
		(dt.timedelta(minutes=5), "minutes"),
		(dt.timedelta(hours=3), "hours"),
		(dt.timedelta(days=2), "days"),
	],
)
def test_unit_follows_the_largest_value(value, expected):
	"""The unit is picked so the biggest value reads as a whole number."""
	assert durations._pick_unit(_frame([value]), "t") == expected


def test_unit_of_an_empty_column_falls_back_to_seconds():
	"""A filtered-away selection still needs a label."""
	assert durations._pick_unit(_frame([]), "t") == "seconds"


@pytest.mark.parametrize(
	("value", "unit", "expected"),
	[
		(dt.timedelta(0), "hours", "0s"),
		(dt.timedelta(milliseconds=350), "seconds", "350ms"),
		# Trimming to whole seconds would call this zero, which it is not.
		(dt.timedelta(milliseconds=350), "hours", "<1s"),
		(dt.timedelta(hours=3, minutes=7, seconds=4, microseconds=500), "hours", "3h 7m 4s"),
		(dt.timedelta(days=2, hours=1), "days", "2d 1h"),
	],
)
def test_hover_text_is_trimmed_to_the_units_resolution(value, unit, expected):
	"""Hover text keeps the precision the axis implies, and never denies a value."""
	frame, _label = durations.to_display(_frame([value]), "t", unit)

	assert frame["t_text"].item() == expected


def test_value_and_label_agree_on_the_unit():
	"""The number plotted and the unit in the axis title come from one call."""
	frame, label = durations.to_display(_frame([dt.timedelta(hours=3)]), "t", "hours")

	assert label == "<b>Time [h]</b>"
	assert frame["t"].item() == pytest.approx(3.0)


# --- scope -------------------------------------------------------------------


def scoped_proportion(context, scope, rows, phase_seconds: float = 100.0) -> float:
	"""proportion_together for one pair at one scope, from (position, seconds) rows."""
	phase_enum = pl.Enum(["light_phase", "dark_phase"])
	pairwise = pl.DataFrame(
		{
			"phase": pl.Series(["light_phase"] * len(rows), dtype=phase_enum),
			"day": pl.Series([1] * len(rows), dtype=pl.UInt16),
			"phase_count": pl.Series([1] * len(rows), dtype=pl.UInt16),
			"hour": pl.Series([0] * len(rows), dtype=pl.UInt8),
			"position": pl.Series([row[0] for row in rows], dtype=pl.Categorical),
			"animal_id": pl.Series([ANIMALS[0]] * len(rows), dtype=pl.Enum(ANIMALS)),
			"animal_id_2": pl.Series([ANIMALS[1]] * len(rows), dtype=pl.Enum(ANIMALS)),
			"time_together": pl.Series(
				[dt.timedelta(seconds=row[1]) for row in rows], dtype=pl.Duration("us")
			),
		}
	)
	phase_durations = pl.DataFrame(
		{
			"phase": pl.Series(["light_phase"], dtype=phase_enum),
			"phase_count": pl.Series([1], dtype=pl.UInt16),
			"duration": pl.Series([dt.timedelta(seconds=phase_seconds)], dtype=pl.Duration("us")),
		}
	)
	time_grid = pairwise.select("phase", "day", "phase_count", "hour").unique()
	scoped = copy.replace(
		context,
		_loaded={
			"pairwise_meetings": pairwise,
			"phase_durations": phase_durations,
			"time_grid": time_grid,
		},
	)

	return (
		prepare._proportion_together(scoped, scope)
		.filter(pl.col("animal_id") == ANIMALS[0], pl.col("animal_id_2") == ANIMALS[1])
		.collect()["proportion_together"]
		.item()
	)


def test_a_mean_over_days_counts_the_days_with_no_chasing(context):
	"""chasings_df stores only the cells a chase happened in; the empty days are zeros."""
	phase_enum = pl.Enum(["light_phase", "dark_phase"])
	time_grid = pl.DataFrame(
		{
			"phase": pl.Series(["light_phase"] * 3, dtype=phase_enum),
			"day": pl.Series([1, 2, 3], dtype=pl.UInt16),
			"phase_count": pl.Series([1, 3, 5], dtype=pl.UInt16),
			"hour": pl.Series([12] * 3, dtype=pl.UInt8),
		}
	)
	chasings = time_grid.head(1).with_columns(
		chaser=pl.Series([ANIMALS[0]], dtype=pl.Enum(ANIMALS)),
		chased=pl.Series([ANIMALS[1]], dtype=pl.Enum(ANIMALS)),
		position=pl.Series(["tunnel_1_a"], dtype=pl.Categorical),
		chasings=pl.Series([6], dtype=pl.UInt32),
	)
	sparse = copy.replace(context, _loaded={"chasings_df": chasings, "time_grid": time_grid})

	matrix = prepare.prep_directed_heatmap(
		sparse,
		(1, 3),
		["light_phase"],
		"mean",
		"day",
		list(ANIMALS),
		"chasings_df",
		"chasings",
		"chaser",
		"chased",
	)

	assert matrix[1, 0] == pytest.approx(2.0)  # 6 chases over 3 days, row chased, column chaser
	assert matrix[0, 1] == 0


def test_tunnels_are_the_positions_that_are_neither_cages_nor_undefined(context):
	"""Tunnels are derived, so a context never has to state them twice."""
	assert context.tunnels == ["tunnel_1", "tunnel_2"]


@pytest.mark.parametrize(
	("scope", "expected"),
	[
		("cages", ["cage_1", "cage_2"]),
		("tunnels", ["tunnel_1", "tunnel_2"]),
		("all", ["cage_1", "cage_2", "tunnel_1", "tunnel_2"]),
	],
)
def test_scope_never_selects_the_undefined_sentinel(context, scope, expected):
	"""``undefined`` is not a place, so no scope may attribute a measure to it."""
	assert context.scope_positions(scope) == expected


@pytest.mark.parametrize(
	("scope", "expected"),
	[
		("cages", {"cage_1"}),
		("tunnels", {"tunnel_1"}),
		("all", {"cage_1", "tunnel_1", "undefined"}),
	],
)
def test_activity_scope_keeps_undefined_only_at_all(context, scope, expected):
	"""Activity shows where time went unplaced, so ``all`` keeps the sentinel."""
	positions = ["cage_1", "tunnel_1", "undefined"]
	activity_df = pl.DataFrame(
		{
			"phase": ["light_phase"] * 3,
			"day": [1] * 3,
			"animal_id": [ANIMALS[0]] * 3,
			"position": pl.Series(positions, dtype=pl.Categorical),
			"visits_to_position": [1] * 3,
			"time_in_position": pl.Series([dt.timedelta(seconds=10)] * 3, dtype=pl.Duration("us")),
		}
	)
	scoped = copy.replace(context, _loaded={**context._loaded, "activity_df": activity_df})

	figure = PlotRegistry.build("activity-bar", scoped, scope=scope)

	assert {x for trace in figure["data"] for x in trace["x"]} == expected


@pytest.mark.parametrize("plot", ["activity-bar", "time-alone-bar"])
@pytest.mark.parametrize("agg", ["sum", "mean"])
def test_position_ticks_name_their_own_bars(context, plot, agg):
	"""A tunnel named ``T1`` sorts before ``cage_1``, so the data arrives tunnel-first."""
	activity_df = pl.DataFrame(
		{
			"phase": ["light_phase"] * 2,
			"day": [1] * 2,
			"animal_id": [ANIMALS[0]] * 2,
			"position": pl.Series(["T1", "cage_1"], dtype=pl.Categorical),
			"visits_to_position": [1, 5],
			"time_in_position": pl.Series([dt.timedelta(seconds=10)] * 2, dtype=pl.Duration("us")),
			"time_alone": pl.Series([dt.timedelta(seconds=1)] * 2, dtype=pl.Duration("us")),
		}
	)
	scoped = copy.replace(
		context,
		cages=["cage_1"],
		positions=["cage_1", "T1", "undefined"],
		_loaded={**context._loaded, "activity_df": activity_df},
	)

	axis = PlotRegistry.build(plot, scoped, scope="all", agg=agg)["layout"]["xaxis"]

	assert axis["ticktext"] == plot_factory._tick_labels(
		[axis["categoryarray"][i] for i in axis["tickvals"]]
	)


def test_unknown_scope_is_rejected(context):
	"""A typo names a scope that would silently select nothing."""
	with pytest.raises(ValueError, match="scope must be one of"):
		context.scope_positions("cage")


def test_proportion_together_counts_only_the_scoped_positions(context):
	"""The scoped value is time together over phase duration, summed within the scope.

	At ``scope="cages"`` this is the arithmetic ``incohort_sociability`` stores (20/100),
	which is what lets the two agree while only one of them can see tunnels.
	"""
	rows = [("cage_1", 20.0), ("tunnel_1", 50.0)]

	assert scoped_proportion(context, "cages", rows) == pytest.approx(0.2)
	assert scoped_proportion(context, "tunnels", rows) == pytest.approx(0.5)
	assert scoped_proportion(context, "all", rows) == pytest.approx(0.7)


@pytest.mark.parametrize(
	("plot", "expected", "default"),
	[
		("time-alone-bar", ("all", "cages", "tunnels"), "all"),
		("cage-preference", ("all", "cages", "tunnels"), "all"),
		("cohort-heatmap", ("all", "cages", "tunnels"), "all"),
		("social-stability", ("all", "cages", "tunnels"), "all"),
		("network-sociability", ("all", "cages", "tunnels"), "all"),
		("cage-preference-evolution", ("cages", "tunnels"), "cages"),
		("sociability-heatmap", ("cages", "tunnels"), "cages"),
	],
)
def test_faceted_plots_offer_no_all_scope(plot, expected, default):
	"""A faceted heatmap's panels share one colour axis, so it shows one kind at a time.

	Cage dwell runs to hours and tunnel dwell to seconds, so mixing them on that shared
	scale renders the tunnel panels a uniform dark. Which plots take ``FacetScope`` and
	which take ``Scope`` is carried by the annotation alone, and a typo there would
	silently restore the unreadable option - so the split is pinned rather than reviewed.
	A faceted plot keeps ``"cages"`` as its default; every other scoped plot now opens
	on ``"all"``.
	"""
	scope = next(option for option in PlotRegistry.spec(plot).options if option.name == "scope")

	assert scope.choices == expected
	assert scope.default == default


# --- colours -----------------------------------------------------------------


def test_only_varying_attributes_are_offered(context):
	"""A cohort with one genotype should not offer to colour by genotype."""
	available = animals_module.available_attributes(context)

	assert available[0] == "animal_id"
	assert "sex" in available
	assert "treatment" in available
	assert "genotype" not in available


def test_animal_colours_do_not_depend_on_the_animal_column(context):
	"""A tag keeps its colour whether the table calls it animal_id, chaser or winner."""
	base = animals_module.resolve_colors(context, "animal_id")
	chaser = animals_module.resolve_colors(context, "animal_id", animal_column="chaser")

	assert base.colors == chaser.colors
	assert chaser.column == "chaser"


def test_attribute_colours_are_shared_within_a_group(context):
	"""Two animals of the same sex get one colour, and the legend one entry."""
	mapping = animals_module.resolve_colors(context, "sex")

	assert mapping.categories == ["F", "M"]
	assert mapping.by_animal["0035A"] == mapping.by_animal["0035B"]
	assert mapping.by_animal["0035A"] != mapping.by_animal["0035C"]


def test_unknown_colour_column_is_rejected(context):
	"""A typo names a column that does not exist, so it fails at the call."""
	with pytest.raises(ValueError, match="color_by"):
		animals_module.resolve_colors(context, "colour")


def test_ordering_stays_in_cohort_order(context):
	"""Colouring by an attribute recolours the plot; it never reorders the animals."""
	assert animals_module.resolve_colors(context, "sex").order == list(context.animal_ids)


def test_legend_collapses_to_one_entry_per_group(context):
	"""Per-animal traces stay, but the legend names each group once."""
	mapping = animals_module.resolve_colors(context, "sex")
	figure = Figure(data=[{"type": "scatter", "name": animal} for animal in ANIMALS], layout={})

	animals_module.collapse_legend(figure, mapping)

	assert [trace["name"] for trace in figure["data"]] == ["M", "M", "F", "F"]
	assert [trace["showlegend"] for trace in figure["data"]] == [True, False, True, False]
	# and the categories' colours are declared, for the recording page's palette swap
	assert figure["layout"]["colorway"] == list(mapping.colors.values())


def test_labelling_by_subject_renames_without_recolouring(context):
	"""Label and colour are independent: an animal keeps its colour under either name."""
	tags = animals_module.resolve_colors(context, "animal_id")
	names = animals_module.resolve_colors(context, "animal_id", label_by="subject_name")
	figure = Figure(data=[{"type": "scatter", "name": animal} for animal in ANIMALS], layout={})

	animals_module.collapse_legend(figure, names)

	assert names.by_animal == tags.by_animal
	assert [trace["name"] for trace in figure["data"]] == ["m1", "m2", "m3", "m4"]
	assert names.legend_title == "<b>Subject name</b>"


def test_a_shared_subject_name_keeps_its_rows_apart(context):
	"""A repeated axis label would merge two animals' heatmap rows into one."""
	cohort = context.animals.with_columns(subject_name=pl.Series(["m1", "m1", "m3", "m4"]))
	shared = copy.replace(context, _loaded={**context._loaded, "animals": cohort})

	assert animals_module.animal_labels(shared, "subject_name") == [
		"m1 (0035A)",
		"m1 (0035B)",
		"m3",
		"m4",
	]


def test_every_colour_by_plot_can_label_by_subject():
	"""The Label control reaches every plot the Animals-by control does."""
	for name in PlotRegistry.list_available():
		options = {option.name for option in PlotRegistry.spec(name).options}
		assert "color_by" not in options or "label_by" in options, name


# --- registry ----------------------------------------------------------------


@pytest.fixture
def clean_registry():
	"""Register throwaway plots without leaking them into other tests."""
	saved = dict(PlotRegistry._specs)
	yield PlotRegistry
	PlotRegistry._specs.clear()
	PlotRegistry._specs.update(saved)


def test_option_without_a_default_is_rejected(clean_registry):
	"""Every option needs a default so a plot can be built with no arguments."""
	with pytest.raises(TypeError, match="no default"):

		@clean_registry.register("broken", title="Broken", info="", requires=())
		def broken(context, *, mode) -> Figure:
			"""Broken."""
			return Figure()


def test_default_outside_its_literal_is_rejected(clean_registry):
	"""A default that no choice allows would fail on every build, so it fails here."""
	with pytest.raises(ValueError, match="not one of"):

		@clean_registry.register("broken", title="Broken", info="", requires=())
		def broken(context, *, mode: Literal["a", "b"] = "c") -> Figure:
			"""Broken."""
			return Figure()


def test_resolver_for_an_unknown_option_is_rejected(clean_registry):
	"""A misspelt resolver key would silently never apply."""
	with pytest.raises(TypeError, match="unknown"):

		@clean_registry.register(
			"broken",
			title="Broken",
			info="",
			requires=(),
			dynamic_choices={"colour_by": lambda c: ["x"]},
		)
		def broken(context, *, color_by: str = "animal_id") -> Figure:
			"""Broken."""
			return Figure()


def test_literal_and_resolver_on_one_option_is_rejected(clean_registry):
	"""The annotation would be discarded, which is certainly a mistake."""
	with pytest.raises(TypeError, match="Literal"):

		@clean_registry.register(
			"broken",
			title="Broken",
			info="",
			requires=(),
			dynamic_choices={"mode": lambda c: ["x"]},
		)
		def broken(context, *, mode: Literal["a", "b"] = "a") -> Figure:
			"""Broken."""
			return Figure()


def test_type_only_annotation_does_not_block_registration(clean_registry):
	"""A builder may annotate with a name imported only for type checking."""

	@clean_registry.register("demo", title="Demo", info="", requires=())
	def demo(context: TypeOnlyContext, *, mode: Literal["a", "b"] = "a") -> Figure:  # noqa: F821
		"""Demo."""
		return Figure()

	assert clean_registry.spec("demo").options[0].choices == ("a", "b")


def test_resolved_default_reaches_the_builder(clean_registry, context):
	"""A narrowed option must pass its resolved value, not its declared one."""
	seen = {}

	@clean_registry.register(
		"demo",
		title="Demo",
		info="",
		requires=("animals",),
		dynamic_choices={"color_by": animals_module.available_attributes},
	)
	def demo(context, *, color_by: str = "genotype") -> Figure:
		"""Demo."""
		seen["color_by"] = color_by
		return Figure()

	clean_registry.build("demo", context)

	# genotype does not vary in this cohort, so the resolver drops it.
	assert seen["color_by"] == "animal_id"


def test_value_outside_its_choices_is_rejected(clean_registry, context):
	"""The choices a spec advertises are enforced, not merely documented."""

	@clean_registry.register("demo", title="Demo", info="", requires=("animals",))
	def demo(context, *, agg: Literal["sum", "mean"] = "sum") -> Figure:
		"""Demo."""
		return Figure()

	with pytest.raises(ValueError, match="must be one of"):
		clean_registry.build("demo", context, agg="median")


def test_missing_required_table_names_the_plot(clean_registry, context):
	"""A plot whose step has not run says so, rather than failing deep inside polars."""

	@clean_registry.register("demo", title="Demo", info="", requires=("chasings_df",))
	def demo(context) -> Figure:
		"""Demo."""
		return Figure()

	with pytest.raises(ValueError, match="chasings_df"):
		clean_registry.build("demo", context)


def test_every_plot_explains_itself_briefly():
	"""The card's info popup has room for a few sentences, not an essay."""
	for name in PlotRegistry.list_available():
		words = len(PlotRegistry.spec(name).info.split())
		assert 0 < words <= 50, name


def test_unit_option_flattens_its_literal_union():
	"""``unit: Unit | Literal["auto"]`` is a union of two Literals, not one."""
	option = next(o for o in PlotRegistry.spec("time-alone-bar").options if o.name == "unit")

	assert set(option.choices) == {"seconds", "minutes", "hours", "days", "auto"}
	assert option.default == "auto"


def test_phase_type_resolves_from_the_cohorts_own_phases(context):
	"""A multi-select option keeps its list default, narrowed to what still applies."""
	described = PlotRegistry.spec("time-alone-bar").options_for(context)
	option = next(o for o in described if o.name == "phase_type")

	assert option.choices == ("light_phase", "dark_phase")
	assert option.default == ["light_phase", "dark_phase"]


def test_hours_range_is_absent_from_whole_day_plots():
	"""Rankings, incohort_sociability and the phase_durations normaliser have no hour column."""
	for name in ("ranking-line", "cohort-heatmap", "social-stability", "network-sociability"):
		options = {option.name for option in PlotRegistry.spec(name).options}
		assert "hours_range" not in options, name

	for name in ("activity-bar", "activity-line", "metrics-polar-line"):
		options = {option.name for option in PlotRegistry.spec(name).options}
		assert "hours_range" in options, name


def test_hours_range_narrows_the_hourly_scaffold(context):
	"""A narrowed hours window drops those hours instead of zero-filling them."""
	frame = pl.DataFrame(
		{
			"animal_id": pl.Series(["0035A"], dtype=pl.Enum(ANIMALS)),
			"day": pl.Series([1], dtype=pl.Int16),
			"hour": pl.Series([5], dtype=pl.Int8),
		}
	)

	with_table = copy.replace(context, _loaded={"t": frame})
	full = prepare.prep_count_line(with_table, (1, 1), "day", "hour", "t", "animal_id", pl.len())
	narrowed = prepare.prep_count_line(
		with_table, (1, 1), "day", "hour", "t", "animal_id", pl.len(), hours_range=(3, 6)
	)

	assert full["hour"].unique().sort().to_list() == list(range(24))
	assert narrowed["hour"].unique().sort().to_list() == [3, 4, 5, 6]


def test_count_line_along_days_folds_the_hours_away(context):
	"""Each day totals its hours, and its mean and SEM run over those hours."""
	frame = pl.DataFrame(
		{
			"animal_id": pl.Series(["0035A"] * 3, dtype=pl.Enum(ANIMALS)),
			"day": pl.Series([1, 1, 2], dtype=pl.Int16),
			"hour": pl.Series([5, 5, 6], dtype=pl.Int8),
		}
	)

	with_table = copy.replace(context, _loaded={"t": frame})
	line = prepare.prep_count_line(
		with_table, (1, 2), "day", "day", "t", "animal_id", pl.len(), hours_range=(5, 6)
	).filter(pl.col("animal_id") == "0035A")

	assert line["day"].to_list() == [1, 2]
	assert line["total"].to_list() == [2, 1]
	assert line["mean"].to_list() == [1.0, 0.5]
	# Hours (2, 0) and (0, 1): std sqrt(2) and sqrt(0.5), over the root of 2 hours.
	assert line["sem"].to_list() == pytest.approx([1.0, 0.5])


def test_line_side_panel_follows_the_legend_and_sits_on_the_zero(context):
	"""Rows group under their legend entry, and the lowest row's bottom edge meets the line's 0."""
	frame = pl.DataFrame(
		{
			"animal_id": pl.Series(ANIMALS * 2, dtype=pl.Enum(ANIMALS)),
			"hour": [0] * 4 + [1] * 4,
			"total": [1, 2, 3, 4, 5, 6, 7, 8],
		}
	)
	no_spans = pl.DataFrame(schema={"event": pl.String, "position": pl.String})
	# Treatment runs vehicle, drug, vehicle, drug, so the legend reads vehicle then drug.
	mapping = animals_module.resolve_colors(context, "treatment")

	figure = plot_factory.plot_count_line(
		frame, mapping, "activity", "hour", "sum", context.phases, no_spans
	)
	side = [trace for trace in figure["data"] if trace.get("xaxis") == "x2"]
	layout = figure["layout"]

	assert [trace["name"] for trace in side] == ["0035A", "0035C", "0035B", "0035D"]
	assert [trace["x"][0] for trace in side] == [6, 10, 8, 12]

	(line_low, line_high), (bottom, top) = layout["yaxis"]["range"], layout["yaxis2"]["range"]
	lowest_edge = len(side) - 1 + plot_factory._SIDE_WIDTH / 2
	assert (lowest_edge - bottom) / (top - bottom) == pytest.approx(
		-line_low / (line_high - line_low)
	)


def test_polar_z_scores_within_the_selection_however_it_is_narrowed(context):
	"""Phase, hours and days all narrow the rows the z-scores are centred on."""
	rows = [
		(animal, day, hour, "dark_phase" if hour >= 12 else "light_phase", 2 * day - (hour < 12))
		for animal in ANIMALS[:2]
		for day in (1, 2)
		for hour in range(24)
	]
	frame = pl.DataFrame(
		rows, schema=["animal_id", "day", "hour", "phase", "phase_count"], orient="row"
	).with_columns(
		metric=pl.lit("activity"),
		value=pl.col("hour").cast(pl.Float64)
		* pl.col("day")
		* (1 + (pl.col("animal_id") == "0035B")),
		exposure=pl.lit(1.0),
	)
	with_table = copy.replace(context, _loaded={"feature_df": frame})
	phases = ["light_phase", "dark_phase"]

	by_phase = prepare.prep_polar(with_table, (1, 1), ["dark_phase"], "day")
	by_hours = prepare.prep_polar(with_table, (1, 1), phases, "day", hours_range=(12, 23))

	assert by_phase.equals(by_hours)
	assert by_phase["mean"].sum() == pytest.approx(0)


# --- events ------------------------------------------------------------------


def bout_cells(cells: list[tuple[str, str | None, int, int]]) -> pl.DataFrame:
	"""An event_bouts table of ``(event, position, day, hour)`` cells."""
	return pl.DataFrame(
		cells,
		schema={
			"event": pl.Enum(["injection", "social"]),
			"position": pl.String,
			"day": pl.UInt16,
			"hour": pl.UInt8,
		},
		orient="row",
	).with_columns(
		start=pl.lit(dt.datetime(2023, 5, 24), pl.Datetime(time_zone="Europe/Warsaw")),
		end=pl.lit(dt.datetime(2023, 5, 25), pl.Datetime(time_zone="Europe/Warsaw")),
	)


def spans(
	context: PlotContext,
	bouts: pl.DataFrame | None,
	x: Literal["datetime", "hour", "day", "phase_count"] = "hour",
) -> list[tuple]:
	"""Event spans over days 1-2 of the window, sorted for comparison."""
	with_bouts = copy.replace(context, _loaded={"event_bouts": bouts} if bouts is not None else {})

	return (
		prepare.prep_event_spans(with_bouts, (1, 2), "day", x)
		.sort("event", "x0")
		.select("event", "position", "x0", "x1")
		.rows()
	)


def test_a_bout_spans_the_part_of_its_day_it_ran(context):
	"""On the day axis an hour-long bout spans that hour of its day, not the whole day.

	A bout running past the window is cut at the window's edge.
	"""
	origin = dt.datetime(2023, 5, 24, 13, tzinfo=ZoneInfo("Europe/Warsaw"))
	hours = [
		("injection", 1, 18, 18, 19),
		*[("social", day, hour, 46, 52) for day, hour in [(2, 22), (2, 23), (3, 0)]],
	]
	bouts = pl.DataFrame(
		[
			(
				event,
				None,
				day,
				hour,
				origin + dt.timedelta(hours=on),
				origin + dt.timedelta(hours=off),
			)
			for event, day, hour, on, off in hours
		],
		schema={
			"event": pl.Enum(["injection", "social"]),
			"position": pl.String,
			"day": pl.UInt16,
			"hour": pl.UInt8,
			"start": pl.Datetime("us", "Europe/Warsaw"),
			"end": pl.Datetime("us", "Europe/Warsaw"),
		},
		orient="row",
	)
	timed = copy.replace(context, experiment_start=origin)

	(injection, social) = spans(timed, bouts, "day")

	assert injection[:2] == ("injection", None)
	assert injection[2:] == pytest.approx((0.5 + 18 / 24, 0.5 + 19 / 24))
	assert social[:2] == ("social", None)
	assert social[2:] == pytest.approx((0.5 + 46 / 24, 2.5))


def test_phase_onsets_are_measured_from_the_start_onset():
	"""The hour axis counts from the start_from onset, so the onset lines must too."""
	recording = strategies.analysis_recording(
		start="2023-05-24 13:00:00",
		phases={"light_phase": dt.time(1, 0), "dark_phase": dt.time(13, 0)},
		start_from="dark_phase",
		end_with="light_phase",
	)

	assert PlotContext.from_recording(recording).phases == {"light_phase": 12.0, "dark_phase": 0.0}


def test_only_the_phase_switch_gets_a_line():
	"""The hour axis opens on the start phase, so its onset is the axis edge."""
	figure = Figure(data=[], layout={})
	no_hours = pl.Series("hour", [], pl.Int8)
	plot_factory._phase_markers(figure, {"light_phase": 13.0, "dark_phase": 0.0}, no_hours)

	assert [shape["x0"] for shape in figure["layout"]["shapes"]] == [13.0]
	assert [shape["line"]["width"] for shape in figure["layout"]["shapes"]] == [1.5]
	assert "annotations" not in figure["layout"]


def test_consecutive_bins_merge_into_one_span(context):
	"""The same hour on several days folds onto the one hour axis."""
	bouts = bout_cells(
		[("social", "cage_1", 1, 13), ("social", "cage_1", 1, 14), ("social", "cage_1", 2, 13)]
	)

	assert spans(context, bouts) == [("social", "cage_1", 12.5, 14.5)]


def test_span_wrapping_past_midnight_splits_in_two(context):
	bouts = bout_cells([("injection", None, 1, 23), ("injection", None, 2, 0)])

	assert spans(context, bouts) == [
		("injection", None, -0.5, 0.5),
		("injection", None, 22.5, 23.5),
	]


def test_day_axis_spans_every_day_touched(context):
	bouts = bout_cells([("social", None, 1, 23), ("social", None, 2, 0)])

	assert spans(context, bouts, "day") == [("social", None, 0.5, 2.5)]


def test_bins_outside_the_window_are_dropped(context):
	assert spans(context, bout_cells([("social", None, 3, 5)])) == []


def test_recording_analysed_before_events_existed_has_no_spans(context):
	assert spans(context, None) == []


def test_datetime_axis_spans_the_bout_on_the_wall_clock(context):
	"""Plotly serializes a trace's datetimes without their zone, so spans drop it too."""
	start = dt.datetime(2023, 5, 24, 23, 30, tzinfo=ZoneInfo("Europe/Warsaw"))
	end = dt.datetime(2023, 5, 25, 0, 30, tzinfo=ZoneInfo("Europe/Warsaw"))
	bouts = bout_cells([("injection", None, 1, 23), ("injection", None, 2, 0)]).with_columns(
		start=pl.lit(start), end=pl.lit(end)
	)

	assert spans(context, bouts, "datetime") == [
		("injection", None, dt.datetime(2023, 5, 24, 23, 30), dt.datetime(2023, 5, 25, 0, 30))
	]


def test_faceted_heatmap_stacks_one_panel_per_row_in_order():
	"""A 2x2 wrap swapped titles between rows; one column keeps facet and row aligned."""
	heatmap = Heatmap(
		values=np.arange(3 * 2 * 4).reshape(3, 2, 4),
		text=None,
		label="Time spent",
		x=list(range(4)),
		y=["a", "b"],
		facets=["cage_1", "cage_2", "cage_3"],
	)

	figure = plot_factory._faceted_heatmap(heatmap, "T", "X", "Y", ("Hour", "Animal ID"))

	layout = figure["layout"]
	titles = [note["text"] for note in layout["annotations"]]
	assert titles == ["<b>Cage 1</b>", "<b>Cage 2</b>", "<b>Cage 3</b>"]
	assert [trace["yaxis"] for trace in figure["data"]] == ["y", "y2", "y3"]
	assert [decoded(trace["z"]).tolist() for trace in figure["data"]] == [
		values.tolist() for values in heatmap.values
	]
	axes = [axis for name, axis in layout.items() if name.startswith(("xaxis", "yaxis"))]
	assert all(axis["showspikes"] and axis["spikemode"] == "across" for axis in axes)


def test_faceted_grid_heatmap_lays_panels_out_in_one_row():
	"""Pair matrices are square, so a wrapped grid left dead space under every row."""
	heatmap = Heatmap(
		values=np.arange(4 * 2 * 2).reshape(4, 2, 2),
		text=None,
		label="Time together",
		x=["a", "b"],
		y=["a", "b"],
		facets=["cage_1", "cage_2", "cage_3", "cage_4"],
	)

	figure = plot_factory._faceted_heatmap(heatmap, "T", "", "", ("X", "Y"), square=True, grid=True)

	assert [trace["xaxis"] for trace in figure["data"]] == ["x", "x2", "x3", "x4"]
	domains = {
		tuple(figure["layout"][trace["yaxis"].replace("y", "yaxis")]["domain"])
		for trace in figure["data"]
	}
	assert len(domains) == 1


@pytest.mark.parametrize(("grid", "rows", "cols"), [(False, 4, 2), (True, 2, 4)])
def test_faceted_heatmap_wraps_past_four_panels_per_line(grid, rows, cols):
	"""An eight-cage habitat squeezed every panel into one column or one row."""
	facets = [f"cage_{index}" for index in range(1, 9)]
	heatmap = Heatmap(
		values=np.zeros((8, 2, 2)), text=None, label="", x=["a", "b"], y=["a", "b"], facets=facets
	)

	figure = plot_factory._faceted_heatmap(heatmap, "T", "", "", ("X", "Y"), grid=grid)

	def domain(trace: dict, axis: str) -> tuple[float, float]:
		return tuple(figure["layout"][trace[f"{axis}axis"].replace(axis, f"{axis}axis")]["domain"])

	assert len({domain(trace, "x") for trace in figure["data"]}) == cols
	assert len({domain(trace, "y") for trace in figure["data"]}) == rows
	# Titles stay over their own panel: each sits centred on its trace's x domain.
	for title, trace in zip(figure["layout"]["annotations"], figure["data"], strict=True):
		assert title["x"] == pytest.approx(sum(domain(trace, "x")) / 2)


def test_sum_line_keeps_its_spline_past_a_thousand_points(context):
	frame = pl.DataFrame(
		{
			"animal_id": pl.Series(ANIMALS * 300, dtype=pl.Enum(ANIMALS)),
			"day": [day for day in range(1, 301) for _ in ANIMALS],
			"total": range(1200),
		}
	)
	mapping = animals_module.resolve_colors(context, "animal_id")
	no_spans = pl.DataFrame(
		schema={
			"event": pl.String,
			"position": pl.String,
			"x0": pl.Float64,
			"x1": pl.Float64,
			"start": pl.Datetime,
			"end": pl.Datetime,
		}
	)

	figure = plot_factory.plot_count_line(
		frame, mapping, "activity", "day", "sum", context.phases, no_spans
	)

	assert {trace["type"] for trace in figure["data"] if "xaxis" not in trace} == {"scatter"}


def test_positioned_event_is_drawn_only_on_its_cages_panel():
	heatmap = Heatmap(
		values=np.zeros((2, 1, 24)),
		text=None,
		label="Time spent",
		x=list(range(24)),
		y=["0035A"],
		facets=["cage_1", "cage_2"],
	)
	bouts = pl.DataFrame(
		{
			"event": ["social", "injection"],
			"position": ["cage_1", None],
			"x0": [12.5, -0.5],
			"x1": [13.5, 0.5],
			"start": [dt.datetime(2023, 5, 24)] * 2,
			"end": [dt.datetime(2023, 5, 25)] * 2,
		},
		schema_overrides={"event": pl.Enum(["injection", "social"])},
	)

	figure = plot_factory.plot_time_spent_per_cage(heatmap, "hour", bouts)
	layout = figure["layout"]
	panels = {
		trace["xaxis"]: facet for facet, trace in zip(heatmap.facets, figure["data"], strict=True)
	}
	drawn = sorted(
		(shape["x0"], panels[shape["xref"]])
		for shape in layout["shapes"]
		if shape.get("name") == "event-span"
	)
	labels = [note["text"] for note in layout["annotations"] if note.get("name") == "event-label"]

	assert drawn == [(-0.5, "cage_1"), (-0.5, "cage_2"), (12.5, "cage_1")]
	assert labels == ["injection", "social"]


def test_one_events_cages_share_a_span_on_a_plot_without_panels(context):
	"""Counterbalanced stimulus sides at the same hours would otherwise stack two labels."""
	ranks = pl.DataFrame({"animal_id": ANIMALS, "day": [1] * 4, "rank": [1.0, 2.0, 3.0, 4.0]})
	bouts = pl.DataFrame(
		{
			"event": ["social", "social"],
			"position": ["cage_3", "cage_1"],
			"x0": [0.5, 0.5],
			"x1": [1.5, 1.5],
			"start": [dt.datetime(2023, 5, 24)] * 2,
			"end": [dt.datetime(2023, 5, 25)] * 2,
		},
		schema_overrides={"event": pl.Enum(["injection", "social"])},
	)
	mapping = animals_module.resolve_colors(context, "animal_id")

	figure = plot_factory.plot_ranking_stability(ranks, mapping, "day", bouts)

	layout = figure["layout"]
	spans = [shape for shape in layout["shapes"] if shape.get("name") == "event-span"]
	(label,) = [note for note in layout["annotations"] if note.get("name") == "event-label"]
	assert len(spans) == 1
	assert (label["text"], label["hovertext"]) == ("social", "social<br>Cage 1<br>Cage 3")


def test_an_events_overlapping_bouts_fuse_into_one_lane_bar():
	"""Overlapping or touching bouts are one bar; a gap starts another; each event, a lane."""
	spans = pl.DataFrame(
		{
			"event": ["social", "social", "social", "injection"],
			"position": ["cage_1", "cage_2", None, None],
			"x0": [0.5, 1.5, 5.5, 0.5],
			"x1": [2.5, 3.5, 6.5, 1.5],
			"start": [dt.datetime(2023, 5, 24)] * 4,
			"end": [dt.datetime(2023, 5, 25)] * 4,
		},
		schema_overrides={"event": pl.Enum(["injection", "social"])},
	)
	figure = Figure(data=[], layout={})

	plot_factory._event_spans(figure, spans)

	lanes = sorted(
		(shape["y0"], shape["x0"], shape["x1"])
		for shape in figure["layout"]["shapes"]
		if shape.get("name") == "event-lane"
	)
	injection, social = sorted({y0 for y0, _, _ in lanes})
	assert [(x0, x1) for y0, x0, x1 in lanes if y0 == injection] == [(0.5, 1.5)]
	assert [(x0, x1) for y0, x0, x1 in lanes if y0 == social] == [(0.5, 3.5), (5.5, 6.5)]


def test_datetime_spans_are_written_like_the_trace_they_mark(context):
	"""Plotly writes a tz-aware trace as naive wall-clock time, so a span must be too.

	Should plotly or polars start writing the offset, spans and data would drift apart.
	"""
	moment = dt.datetime(2023, 5, 24, 13, 0, 30, tzinfo=ZoneInfo("Europe/Warsaw"))
	ranking = pl.DataFrame(
		{"animal_id": [ANIMALS[0]], "datetime": [moment], "ordinal": [1.0], "day": 1, "hour": 13},
		schema={
			"animal_id": pl.String,
			"datetime": pl.Datetime("us", "Europe/Warsaw"),
			"ordinal": pl.Float64,
			"day": pl.Int16,
			"hour": pl.Int8,
		},
	)
	bouts = bout_cells([("injection", None, 1, 13)]).with_columns(
		start=pl.lit(moment), end=pl.lit(moment + dt.timedelta(minutes=10))
	)
	loaded = {"ranking": ranking} | ({"event_bouts": bouts} if bouts is not None else {})
	with_bouts = copy.replace(context, _loaded=loaded)
	mapping = animals_module.resolve_colors(context, "animal_id")

	figure = plot_factory.plot_ranking_line(
		prepare.prep_ranking_over_time(with_bouts, (1, 2), "day"),
		mapping,
		prepare.prep_event_spans(with_bouts, (1, 2), "day", "datetime"),
	)
	payload = json.loads(to_json_plotly(figure))

	assert dt.datetime.fromisoformat(payload["layout"]["shapes"][0]["x0"]) == (
		dt.datetime.fromisoformat(payload["data"][0]["x"][0])
	)


def test_timeline_buckets_take_their_dominant_position_until_zoomed_in(context):
	"""A 2000 s span buckets by the second, so a 0.3 s tunnel crossing folds into the cage
	it was mostly spent in; zoomed to 1.5 s, the buckets are milliseconds and it is back."""
	zone = "Europe/Warsaw"
	noon = dt.datetime(2023, 5, 24, 12, tzinfo=ZoneInfo(zone))
	main_df = pl.DataFrame(
		{
			"animal_id": ["0035A", "0035A", "0035A", "0035B"],
			"position": ["cage_1", "tunnel_1_a", "cage_2", "cage_1"],
			"datetime": [noon + dt.timedelta(seconds=s) for s in (1000, 1000.3, 2000, 2000)],
			"time_spent": [dt.timedelta(seconds=s) for s in (1000, 0.3, 999.7, 2000)],
			"day": [1] * 4,
		},
		schema_overrides={"datetime": pl.Datetime("us", zone)},
	)
	context = copy.replace(context, _loaded={"padded_df": main_df})

	def wall(seconds: float) -> dt.datetime:
		return dt.datetime(2023, 5, 24, 12) + dt.timedelta(seconds=seconds)

	assert prepare.prep_timeline(context, (1, 1), "day").rows() == [
		("0035A", "cage_1", wall(0), wall(1000)),
		("0035A", "cage_2", wall(1000), wall(2000)),
		("0035B", "cage_1", wall(0), wall(2000)),
	]

	zoomed = ("2023-05-24 12:16:39.5", "2023-05-24 12:16:41")
	assert prepare.prep_timeline(context, (1, 1), "day", x_range=zoomed).rows() == [
		("0035A", "cage_1", wall(999.5), wall(1000)),
		("0035A", "tunnel_1", wall(1000), wall(1000.3)),
		("0035A", "cage_2", wall(1000.3), wall(1001)),
		("0035B", "cage_1", wall(999.5), wall(1001)),
	]


def test_timeline_bars_stop_where_the_visits_do_not_at_bucket_edges(context):
	"""The 999 ms buckets of this 1999.75 s span are counted from the epoch, so neither the
	first read, the undefined gap nor the last read falls on an edge; the bars still do."""
	zone = "Europe/Warsaw"
	noon = dt.datetime(2023, 5, 24, 12, tzinfo=ZoneInfo(zone))
	main_df = pl.DataFrame(
		{
			"animal_id": ["0035A"] * 3,
			"position": ["cage_1", "undefined", "cage_2"],
			"datetime": [noon + dt.timedelta(seconds=s) for s in (1000.5, 1200, 2000.25)],
			"time_spent": [dt.timedelta(seconds=s) for s in (1000, 199.5, 800.25)],
			"day": [1] * 3,
		},
		schema_overrides={"datetime": pl.Datetime("us", zone)},
	)
	context = copy.replace(context, _loaded={"padded_df": main_df})

	def wall(seconds: float) -> dt.datetime:
		return dt.datetime(2023, 5, 24, 12) + dt.timedelta(seconds=seconds)

	assert prepare.prep_timeline(context, (1, 1), "day").rows() == [
		("0035A", "cage_1", wall(0.5), wall(1000.5)),
		("0035A", "cage_2", wall(1200), wall(2000.25)),
	]


def test_timeline_extent_keeps_a_leading_undefined_stretch(context):
	"""A recording gap at the window's start draws no bar, but the axis still opens there."""
	zone = "Europe/Warsaw"
	noon = dt.datetime(2023, 5, 24, 12, tzinfo=ZoneInfo(zone))
	padded_df = pl.DataFrame(
		{
			"animal_id": ["0035A"] * 2,
			"position": ["undefined", "cage_1"],
			"datetime": [noon + dt.timedelta(hours=h) for h in (3, 4)],
			"time_spent": [dt.timedelta(hours=3), dt.timedelta(hours=1)],
			"day": [1] * 2,
		},
		schema_overrides={"datetime": pl.Datetime("us", zone)},
	)
	context = copy.replace(context, _loaded={"padded_df": padded_df})

	assert prepare.prep_timeline_extent(context, (1, 1), "day") == (
		"2023-05-24 12:00:00.000000",
		"2023-05-24 16:00:00.000000",
	)


def test_timeline_draws_a_visit_that_began_before_the_window(context):
	"""A stay read on day 1 and again on day 3 still fills day 2, which holds neither read."""
	zone = "Europe/Warsaw"
	noon = dt.datetime(2023, 5, 24, 12, tzinfo=ZoneInfo(zone))
	pieces = [noon + dt.timedelta(days=1, minutes=m) for m in range(2)]
	padded_df = pl.DataFrame(
		{
			"animal_id": ["0035A"] * 2,
			"position": ["cage_1"] * 2,
			"datetime": [piece + dt.timedelta(minutes=1) for piece in pieces],
			"time_spent": [dt.timedelta(minutes=1)] * 2,
			"day": [2] * 2,
		},
		schema_overrides={"datetime": pl.Datetime("us", zone)},
	)
	context = copy.replace(context, _loaded={"padded_df": padded_df})

	assert prepare.prep_timeline(context, (2, 2), "day").rows() == [
		("0035A", "cage_1", dt.datetime(2023, 5, 25, 12), dt.datetime(2023, 5, 25, 12, 2)),
	]


def test_timeline_leaves_an_undefined_gap_empty_between_bars_at_one_position(context):
	"""The 1.4 s gap spends under half of either 1 s bucket at cage_1, so both buckets go
	undefined and the bar breaks, where a vote among defined visits alone bridged it."""
	zone = "Europe/Warsaw"
	noon = dt.datetime(2023, 5, 24, 12, tzinfo=ZoneInfo(zone))
	main_df = pl.DataFrame(
		{
			"animal_id": ["0035A"] * 3,
			"position": ["cage_1", "undefined", "cage_1"],
			"datetime": [noon + dt.timedelta(seconds=s) for s in (1000.3, 1001.7, 2000)],
			"time_spent": [dt.timedelta(seconds=s) for s in (1000.3, 1.4, 998.3)],
			"day": [1] * 3,
		},
		schema_overrides={"datetime": pl.Datetime("us", zone)},
	)
	context = copy.replace(context, _loaded={"padded_df": main_df})

	def wall(seconds: float) -> dt.datetime:
		return dt.datetime(2023, 5, 24, 12) + dt.timedelta(seconds=seconds)

	assert prepare.prep_timeline(context, (1, 1), "day").rows() == [
		("0035A", "cage_1", wall(0), wall(1000)),
		("0035A", "cage_1", wall(1002), wall(2000)),
	]


def test_timeline_ships_wall_clock_strings_and_reads_back_as_dates():
	"""The timeline's x is wall-clock strings - plotly reads epoch numbers in the browser's
	zone - and its y a typed array."""
	zone = "Europe/Warsaw"

	def at(hour: int) -> dt.datetime:
		return dt.datetime(2023, 5, 24, hour, tzinfo=ZoneInfo(zone))

	visits = pl.DataFrame(
		{
			"animal_id": ["0035B", "0035A", "0035A"],
			"position": ["cage_1", "cage_1", "tunnel_1"],
			"start": [at(1), at(2), at(4)],
			"end": [at(2), at(3), at(5)],
		},
		schema_overrides={"start": pl.Datetime("us", zone), "end": pl.Datetime("us", zone)},
	)
	no_spans = pl.DataFrame(schema={"event": pl.String, "position": pl.String})

	figure = plot_factory.plot_timeline(
		prepare.prep_timeline_vertices(visits),
		ANIMALS[:2],
		ANIMALS[:2],
		["cage_1", "tunnel_1"],
		no_spans,
		("2023-05-24 01:00:00", "2023-05-24 05:00:00"),
	)
	payload = json.loads(to_json_plotly(figure))

	assert [(t["name"], t["meta"], t["showlegend"]) for t in payload["data"]] == [
		("cage_1", "0035A", True),
		("cage_1", "0035B", False),
		("tunnel_1", "0035A", True),
		("tunnel_1", "0035B", False),
	]
	x = payload["data"][0]["x"]
	# Start, two interior hover vertices, end, and the None that breaks the line.
	assert len(x) == 3 + 2
	# A zero fraction is left off, which is most of the payload's bytes saved.
	assert (x[0], x[1], x[3], x[4]) == (
		"2023-05-24 02:00:00",
		"2023-05-24 02:20:00",
		"2023-05-24 03:00:00",
		None,
	)
	# Plotly hides a zero-length trace, legend entry and all.
	assert all(trace["x"] for trace in payload["data"])
	assert all("bdata" in trace["y"] for trace in payload["data"])
	assert payload["layout"]["xaxis"]["type"] == "date"


# --- quality -------------------------------------------------------------------


def test_quality_heatmap_pivots_miss_rate_by_animal_and_antenna(context):
	quality = pl.DataFrame(
		{
			"animal_id": [ANIMALS[0], ANIMALS[0], ANIMALS[1], ANIMALS[1]],
			"antenna": [1, 2, 1, 2],
			"correct": [10, 10, 10, 10],
			"interpolated": [0, 2, 0, 0],
			"bad": [0, 3, 2, 0],
			"miss_rate": [0.0, 33.3, 16.7, 0.0],
		}
	)
	with_quality = copy.replace(context, _loaded={"recording_quality": quality})

	matrix, antennas = prepare.prep_quality_heatmap(with_quality, [ANIMALS[0], ANIMALS[1]])

	assert antennas == ["1", "2"]
	assert matrix.tolist() == [[0.0, 33.3], [16.7, 0.0]]


def test_heatmap_rows_follow_the_label(context):
	"""Labelling by subject renames the rows, which stay in cohort order."""
	quality = pl.DataFrame(
		{
			"animal_id": ANIMALS,
			"antenna": [1] * 4,
			"correct": [10] * 4,
			"interpolated": [0] * 4,
			"bad": [0, 1, 2, 3],
			"miss_rate": [0.0, 9.1, 16.7, 23.1],
		}
	)
	with_quality = copy.replace(context, _loaded={**context._loaded, "recording_quality": quality})

	figure = PlotRegistry.build("quality-heatmap", with_quality, label_by="subject_name")

	assert figure["data"][0]["y"] == ["m1", "m2", "m3", "m4"]


def test_a_heatmaps_data_is_one_row_per_cell_under_its_labels(context):
	quality = pl.DataFrame(
		{
			"animal_id": [*ANIMALS, ANIMALS[0]],
			"antenna": [1, 1, 1, 1, 2],
			"correct": [10] * 5,
			"interpolated": [0] * 5,
			"bad": [1, 0, 0, 0, 0],
			"miss_rate": [9.1, 0.0, 0.0, 0.0, 0.0],
		}
	)
	with_quality = copy.replace(context, _loaded={**context._loaded, "recording_quality": quality})

	(data,) = PlotRegistry.data("quality-heatmap", with_quality, label_by="subject_name").values()

	assert data.columns == ["animal", "antenna", "missed_%"]
	assert data.height == 4 * 2
	assert data.filter(antenna="1")["animal"].to_list() == ["m1", "m2", "m3", "m4"]
	assert data.filter(animal="m1", antenna="1")["missed_%"].item() == 9.1


def test_quality_by_antenna_pools_counts_rather_than_averaging_rates():
	"""A pooled share weighs by how much was actually seen, not by cell count."""
	quality = pl.DataFrame(
		{
			"animal_id": ["a", "a", "b", "b"],
			"antenna": [1, 2, 1, 2],
			# Antenna 1: 1 bad of 1001. Antenna 2: 1 bad and 1 interpolated of 4 - a high
			# per-cell rate that must not outweigh antenna 1's much larger sample.
			"correct": [1000, 1, 0, 1],
			"interpolated": [0, 1, 0, 0],
			"bad": [1, 0, 0, 1],
			"miss_rate": [0.1, 50.0, 0.0, 50.0],
		}
	)
	context = PlotContext(
		_loaded={"recording_quality": quality},
		animal_ids=["a", "b"],
		cages=[],
		positions=[],
		phases={},
		days_range=(1, 1),
		phase_range=(1, 1),
		tunnels_map={},
	)

	frame = prepare.prep_quality_by_antenna(context)

	frame = frame.sort("antenna")
	assert frame["bad_share"].to_list() == pytest.approx([1 / 1001 * 100, 1 / 4 * 100])
	assert frame["interpolated_share"].to_list() == pytest.approx([0.0, 1 / 4 * 100])
	shares = frame.select(pl.sum_horizontal(pl.col("^.*_share$")))
	assert shares.to_series().to_list() == pytest.approx([100.0, 100.0])


def test_reads_per_antenna_stacks_every_kind_to_100_percent(context):
	"""One bar per kind, stacked, each in its theme colour."""
	quality = pl.DataFrame(
		{
			"animal_id": ANIMALS[:2],
			"antenna": [1, 1],
			"correct": [8, 9],
			"interpolated": [1, 0],
			"bad": [1, 1],
			"miss_rate": [20.0, 10.0],
		}
	)
	with_quality = copy.replace(context, _loaded={**context._loaded, "recording_quality": quality})

	figure = theme.apply(PlotRegistry.build("quality-antenna", with_quality), "dark")

	bars = figure["data"]
	assert figure["layout"]["barmode"] == "stack"
	assert [bar["name"] for bar in bars] == ["Correct", "Interpolated", "Bad"]
	assert sum(decoded(bar["y"])[0] for bar in bars) == pytest.approx(100.0)
	assert [bar["marker"]["color"] for bar in bars] == list(theme.READ_LOOKS.values())


def test_ranking_distribution_is_the_same_for_a_window_in_days_or_phases(context):
	"""The window only picks which matches count, so its unit must not change the fit."""
	# One row per animal after each match, in replay order; phase 2 has no matches.
	ranking = pl.DataFrame(
		{
			"animal_id": ["0035A", "0035B"] * 3,
			"mu": [26.0, 24.0, 28.0, 22.0, 30.0, 20.0],
			"sigma": [8.0] * 6,
			"day": [1, 1, 2, 2, 2, 2],
			"phase_count": [1, 1, 3, 3, 4, 4],
		}
	)
	context = copy.replace(context, _loaded={"ranking": ranking})

	def fit(days_range, granularity):
		return prepare.prep_ranking_distribution(context, days_range, granularity)

	assert fit((1, 2), "day").equals(fit((1, 4), "phase_count"))
	# The nearest phase with matches after the window must not stand in for its empty end.
	assert fit((1, 1), "day").equals(fit((1, 2), "phase_count"))


def test_ranking_over_time_keeps_only_rating_changes_and_each_animals_last(context):
	"""A match moves just its two animals, so an unchanged rating adds nothing to the step line."""
	ranking = pl.DataFrame(
		{
			"animal_id": ["0035A", "0035B"] * 4,
			"ordinal": [0.0, 0.0, 1.0, 0.0, 1.0, -1.0, 1.0, -1.0],
			"datetime": [dt.datetime(2023, 5, 24, hour) for hour in (1, 1, 2, 2, 3, 3, 4, 4)],
			"day": [1] * 8,
			"hour": [1, 1, 2, 2, 3, 3, 4, 4],
		}
	)
	context = copy.replace(context, _loaded={"ranking": ranking})

	frame = prepare.prep_ranking_over_time(context, (1, 1), "day")

	assert frame.select("animal_id", "hour").rows() == [
		("0035A", 1),
		("0035B", 1),
		("0035A", 2),
		("0035B", 3),
		("0035A", 4),
		("0035B", 4),
	]


# --- network graphs ----------------------------------------------------------


def test_dominance_node_size_is_the_animals_own_ordinal(context):
	"""Sizes are looked up by animal id; the ranking table is in no particular order."""
	animal = pl.Enum(ANIMALS)
	chasings = pl.DataFrame(
		{
			"chaser": pl.Series(ANIMALS[:2], dtype=animal),
			"chased": pl.Series(ANIMALS[1:3], dtype=animal),
			"chasings": [3, 1],
			"day": [1, 1],
		}
	)
	ranking = pl.DataFrame(
		{"animal_id": ANIMALS[::-1], "ordinal": [10.0, 20.0, 30.0, 40.0], "day": [1] * 4}
	)
	context = copy.replace(context, _loaded={"chasings_df": chasings, "ranking": ranking})

	nodes, edges = prepare.prep_network_dominance(context, (1, 1), "day", "circular", 0)

	assert nodes["size"].to_list() == [40.0, 30.0, 20.0, 10.0]
	assert edges.filter(pl.col("source") == pl.col("target")).is_empty()


def test_edge_cutoff_refits_the_layout_without_the_weak_edges():
	"""Weak edges are dropped before the layout, so the nodes move rather than lines hiding."""
	connections = pl.DataFrame(
		{"source": ["a", "b", "a"], "target": ["b", "c", "c"], "chasings": [10.0, 9.0, 1.0]}
	)
	sizes = {"a": 30.0, "b": 20.0, "c": 10.0}

	full_nodes, full_edges = prepare._network_layout(
		connections, "chasings", True, sizes, "spring", 0
	)
	cut_nodes, cut_edges = prepare._network_layout(
		connections, "chasings", True, sizes, "spring", 50
	)

	assert full_edges.height == 3
	assert cut_edges.height == 2
	assert full_nodes["x"].to_list() != cut_nodes["x"].to_list()


def _ranking_context(context, ranking, chasings=None):
	chasings = (
		chasings
		if chasings is not None
		else pl.DataFrame(
			schema={
				"chaser": pl.Enum(ANIMALS),
				"chased": pl.Enum(ANIMALS),
				"chasings": pl.Int64,
				"day": pl.Int64,
			}
		)
	)
	return copy.replace(context, _loaded={"chasings_df": chasings, "ranking": ranking})


def test_dominance_node_size_is_the_rating_at_the_windows_end(context):
	"""A rating earned after the window must not stand in for the one at its close."""
	ranking = pl.DataFrame(
		{
			"animal_id": ["0035A", "0035B"] * 4,
			"ordinal": [1.0, 0.0, 2.0, 0.0, 9.0, 0.0, 20.0, 0.0],
			"day": [1, 1, 2, 2, 4, 4, 5, 5],
		}
	)
	nodes, _ = prepare.prep_network_dominance(
		_ranking_context(context, ranking), (1, 3), "day", "circular", 0
	)

	assert dict(zip(nodes["animal_id"], nodes["size"], strict=True)) == {
		"0035A": 2.0,
		"0035B": 0.0,
		"0035C": 0.0,
		"0035D": 0.0,
	}


def test_dominance_network_without_any_rating_is_drawn_unsized(context):
	ranking = pl.DataFrame(schema={"animal_id": pl.String, "ordinal": pl.Float64, "day": pl.Int64})

	nodes, _ = prepare.prep_network_dominance(
		_ranking_context(context, ranking), (1, 1), "day", "circular", 0
	)

	assert nodes["size"].to_list() == [0.0] * 4


def test_ranking_distribution_reaches_ratings_beyond_the_old_fixed_axis(context):
	ranking = pl.DataFrame(
		{
			"animal_id": ["0035A", "0035B"],
			"mu": [61.0, -11.0],
			"sigma": [4.8, 4.8],
			"day": [1, 1],
		}
	)

	frame = prepare.prep_ranking_distribution(_ranking_context(context, ranking), (1, 1), "day")

	assert frame["ranking"].min() <= -11 - 4 * 4.8 + 1
	assert frame["ranking"].max() >= 61 + 4 * 4.8 - 1


def test_node_hover_names_the_real_rating_not_the_marker_size():
	nodes = pl.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0], "size": [-3.0, 5.0]})

	trace = plot_factory._node_trace(nodes, ["red", "blue"], ["a", "b"], include_ranking=True)

	assert trace["marker"]["size"] == [0.1, 5.0]
	assert trace["hovertext"] == ["Mouse ID: a<br>Ranking: -3.0", "Mouse ID: b<br>Ranking: 5.0"]


def _phenotype(ordinals):
	return pl.DataFrame(
		{
			"animal_id": ANIMALS,
			"subject_name": ["m1", "m2", "m3", "m4"],
			"visits": [10, 20, 30, 40],
			"gregariousness": [0.1, 0.2, 0.3, 0.4],
			"won": [0, 1, 2, 3],
			"ordinal": ordinals,
			"rating_size": [1.0] * 4,
		}
	)


def test_phenotype_annotation_points_at_the_highest_rated_animal(context):
	mapping = animals_module.resolve_colors(context, "animal_id")

	figure = plot_factory.plot_phenotype_map(_phenotype([1.0, 5.0, 2.0, None]), mapping)

	(note,) = figure["layout"]["annotations"]
	assert (note["x"], note["y"]) == (20, 0.2)


def test_phenotype_without_ratings_claims_no_hierarchy(context):
	mapping = animals_module.resolve_colors(context, "animal_id")

	figure = plot_factory.plot_phenotype_map(_phenotype([None] * 4), mapping)

	assert "annotations" not in figure["layout"]


# --- module boundaries -------------------------------------------------------


def test_core_does_not_import_plotting():
	"""The analysis pipeline must be usable without plotly installed."""
	probe = (
		"import sys;"
		"import deepecohab;"
		"import deepecohab.core.antenna_analysis;"
		"import deepecohab.core.recording_pipeline;"
		"print('plotly' in sys.modules, 'networkx' in sys.modules)"
	)
	result = subprocess.run(
		[sys.executable, "-c", probe], capture_output=True, text=True, check=True
	)

	assert result.stdout.strip() == "False False"


def test_importing_the_package_leaves_the_plotly_default_theme_alone():
	"""Importing a package should not restyle every other figure in the session."""
	probe = (
		"import plotly.io as pio;"
		"before = pio.templates.default;"
		"import deepecohab;"
		"print(before == pio.templates.default)"
	)
	result = subprocess.run(
		[sys.executable, "-c", probe], capture_output=True, text=True, check=True
	)

	assert result.stdout.strip() == "True"
