from collections.abc import Sequence
from dataclasses import replace
from typing import Literal

import plotly.graph_objects as go
import polars as pl

from deepecohab.core.data_model import Layout
from deepecohab.plotting import durations, plot_factory, prepare
from deepecohab.plotting.animals import (
	animal_labels,
	available_attributes,
	mean_by_group,
	resolve_colors,
)
from deepecohab.plotting.context import (
	SCOPE_NOUN,
	FacetScope,
	Granularity,
	LabelBy,
	PlotContext,
	Scope,
)
from deepecohab.plotting.durations import Unit
from deepecohab.plotting.registry import PlotRegistry
from deepecohab.plotting.theme import sample_palette

PHASES = ("light_phase", "dark_phase")
BY_COHORT = {"color_by": available_attributes}
#: ``Timeline.phases`` need not carry both keys, so a plot's phase_type choices are
#: resolved from the cohort's own phases rather than assumed to always be the pair.
PHASE_TYPE = {"phase_type": lambda context: list(context.phases)}
BY_COHORT_AND_PHASE = {**BY_COHORT, **PHASE_TYPE}


@PlotRegistry.register(
	"animal-speed",
	title="Tunnel-crossing speed",
	info=(
		"Median crossing speed per animal and day or phase, pooled into one box per tunnel. "
		"Speed uses the configured tunnel distance. Both directions count; only positive "
		"durations up to Max dwell (seconds) contribute."
	),
	requires=("main_df",),
	dynamic_choices=PHASE_TYPE,
)
def animal_speed(
	context: PlotContext,
	*,
	days_range: tuple[int, int] | None = None,
	granularity: Granularity = "day",
	phase_type: Sequence[str] = PHASES,
	hours_range: tuple[int, int] | None = None,
	max_dwell: float = 10,
) -> go.Figure:
	"""Per-tunnel distributions of each animal's daily or per-phase median speed."""
	frame = prepare.prep_speed_box(
		context,
		_window(context, days_range, granularity),
		phase_type,
		granularity,
		hours_range,
		max_dwell,
	)
	return plot_factory.plot_animal_speed(
		frame, context.tunnels, sample_palette(len(context.tunnels)), granularity
	)


@PlotRegistry.register(
	"animal-speed-daily",
	title="Mean tunnel-crossing speed",
	info=(
		"Crossing speeds averaged per animal, day or phase, and hour, then averaged over "
		"the axis folded away. The band shows SEM across observed cells; empty cells stay "
		"null. Speed uses configured tunnel distances and the Max dwell cutoff in seconds."
	),
	requires=("main_df", "animals"),
	dynamic_choices=BY_COHORT_AND_PHASE,
)
def animal_speed_daily(
	context: PlotContext,
	*,
	days_range: tuple[int, int] | None = None,
	granularity: Granularity = "day",
	phase_type: Sequence[str] = PHASES,
	color_by: str = "animal_id",
	label_by: LabelBy = "animal_id",
	hours_range: tuple[int, int] | None = None,
	timescale: Literal["days", "hours"] = "hours",
	max_dwell: float = 10,
	group_mean: bool = False,
) -> go.Figure:
	"""Mean tunnel-crossing speed with SEM, by hour or by day/phase."""
	window = _window(context, days_range, granularity)
	x = granularity if timescale == "days" else "hour"
	frame = prepare.prep_speed_line(
		context, window, phase_type, granularity, x, hours_range, max_dwell
	)
	spans = prepare.prep_event_spans(context, window, granularity, x, hours_range)
	mapping = resolve_colors(context, color_by, group_mean=group_mean, label_by=label_by)
	frame = mean_by_group(frame, mapping, ["mean"])
	return plot_factory.plot_mean_line(frame, mapping, "speed", x, context.phases, spans)


#: The count-line builder each aggregation draws with.
LINE_BY_AGG = {
	"sum": plot_factory.plot_sum_line,
	"mean": plot_factory.plot_mean_line,
}


def _window(
	context: PlotContext,
	days_range: tuple[int, int] | None,
	granularity: Granularity,
) -> tuple[int, int]:
	"""Fall back to the whole recording when no window is selected."""
	return context.axis_range(granularity) if days_range is None else days_range


@PlotRegistry.register(
	"recording-timeline",
	title="Position timeline",
	info=(
		"Each bar is one visit, running from the animal's previous antenna read to the read "
		"that ended it; consecutive reads at the same position merge into one bar. One row per "
		"animal, coloured by position. Undefined positions are left as gaps."
	),
	requires=("main_df", "animals"),
)
def recording_timeline(
	context: PlotContext,
	x_range: tuple[str, str] | None = None,
	*,
	days_range: tuple[int, int] | None = None,
	granularity: Granularity = "day",
	hours_range: tuple[int, int] | None = None,
	label_by: LabelBy = "animal_id",
) -> go.Figure:
	"""Every animal's position over time, as a compact Gantt-style strip.

	``x_range`` is the span a zoom shows, resampled at its own resolution. It sits before
	the ``*`` because it follows the plot's axis, not a control, so it is no card option.
	"""
	window = _window(context, days_range, granularity)
	frame = prepare.prep_timeline(context, window, granularity, hours_range, x_range)
	spans = prepare.prep_event_spans(context, window, granularity, "datetime", hours_range)

	figure = plot_factory.plot_timeline(
		frame, context.animal_ids, context.scope_positions("all"), spans
	)
	return figure.update_yaxes(ticktext=animal_labels(context, label_by))


@PlotRegistry.register(
	"activity-bar",
	title="Activity per position",
	info=(
		"Visits to each position, or time spent there, per animal, summed over the selected "
		"phases and window. Sum draws grouped bars of the window total; mean draws boxes over "
		"the per-day (or per-phase) values, the dashed line marking the mean. Undefined is time "
		"no antenna could place."
	),
	requires=("activity_df", "animals"),
	dynamic_choices=BY_COHORT_AND_PHASE,
)
def activity(
	context: PlotContext,
	*,
	days_range: tuple[int, int] | None = None,
	granularity: Granularity = "day",
	phase_type: Sequence[str] = PHASES,
	metric: Literal["visits", "time"] = "time",
	agg: Literal["sum", "mean"] = "sum",
	scope: Scope = "all",
	color_by: str = "animal_id",
	label_by: LabelBy = "animal_id",
	unit: Unit | Literal["auto"] = "auto",
	hours_range: tuple[int, int] | None = None,
	group_mean: bool = False,
) -> go.Figure:
	"""Visits to each position, or time spent there.

	Quantifies behaviour either by the number of visits to specific locations or
	the total time spent in those locations. ``"all"`` keeps the undefined position,
	so the time no antenna could place stays visible.
	"""
	window = _window(context, days_range, granularity)
	positions = context.positions if scope == "all" else context.scope_positions(scope)
	frame = prepare.prep_activity(
		context,
		window,
		phase_type,
		granularity,
		agg,
		positions,
		{"visits": "visits_to_position", "time": "time_in_position"},
		hours_range,
	)
	mapping = resolve_colors(context, color_by, group_mean=group_mean, label_by=label_by)
	frame = mean_by_group(frame, mapping, ["visits", "time"])
	frame, label = durations.to_display(frame, "time", unit, "Time spent")

	return plot_factory.plot_activity(
		frame, positions, mapping, metric, agg, granularity, label, SCOPE_NOUN[scope]
	)


@PlotRegistry.register(
	"time-alone-bar",
	title="Time spent alone",
	info=(
		"Time each animal spent at a position with no other animal there, summed per position "
		"over the selected phases and window. Sum draws grouped bars of the total; mean draws "
		"boxes over the per-day (or per-phase) totals."
	),
	requires=("activity_df", "animals"),
	dynamic_choices=BY_COHORT_AND_PHASE,
)
def time_alone(
	context: PlotContext,
	*,
	days_range: tuple[int, int] | None = None,
	granularity: Granularity = "day",
	phase_type: Sequence[str] = PHASES,
	agg: Literal["sum", "mean"] = "sum",
	scope: Scope = "all",
	color_by: str = "animal_id",
	label_by: LabelBy = "animal_id",
	unit: Unit | Literal["auto"] = "auto",
	hours_range: tuple[int, int] | None = None,
	group_mean: bool = False,
) -> go.Figure:
	"""Time each animal spent without any other animal present.

	Shows the duration each animal spent alone, segmented by the position where that
	happened - cages, tunnels, or both.
	"""
	window = _window(context, days_range, granularity)
	positions = context.scope_positions(scope)
	frame = prepare.prep_activity(
		context,
		window,
		phase_type,
		granularity,
		agg,
		positions,
		{"time_alone": "time_alone"},
		hours_range,
	)
	mapping = resolve_colors(context, color_by, group_mean=group_mean, label_by=label_by)
	frame = mean_by_group(frame, mapping, ["time_alone"])
	frame, label = durations.to_display(frame, "time_alone", unit, "Time alone")

	return plot_factory.plot_time_alone(
		frame, positions, mapping, agg, granularity, label, SCOPE_NOUN[scope]
	)


@PlotRegistry.register(
	"cage-preference",
	title="Position preference",
	info=(
		"Time each animal spent in each position, summed per day (or phase) over the selected "
		"phases. Each box pools every animal's per-day totals, so it shows how the cohort as a "
		"whole split its time; the dashed line is the mean, points are outliers."
	),
	requires=("activity_df",),
	dynamic_choices=PHASE_TYPE,
)
def cage_preference(
	context: PlotContext,
	*,
	days_range: tuple[int, int] | None = None,
	granularity: Granularity = "day",
	phase_type: Sequence[str] = PHASES,
	scope: Scope = "all",
	unit: Unit | Literal["auto"] = "auto",
	hours_range: tuple[int, int] | None = None,
) -> go.Figure:
	"""Distribution of time the cohort spends in each position."""
	window = _window(context, days_range, granularity)
	positions = context.scope_positions(scope)
	frame, label = prepare.prep_cage_preference(
		context, window, phase_type, granularity, positions, unit, hours_range
	)

	return plot_factory.plot_cage_preference(
		frame,
		positions,
		sample_palette(len(positions)),
		granularity,
		label,
		SCOPE_NOUN[scope],
	)


@PlotRegistry.register(
	"cage-preference-evolution",
	title="Position preference over time",
	info=(
		"Time each animal spent in each cage (or tunnel), drawn as one heatmap per position "
		"with animals as rows. Columns step through the window's days or phases, or fold it "
		"onto the 24 hours of the day. Sum totals each cell's time; mean averages the hourly "
		"values it gathers."
	),
	requires=("activity_df", "animals"),
)
def cage_preference_evolution(
	context: PlotContext,
	*,
	days_range: tuple[int, int] | None = None,
	granularity: Granularity = "day",
	timescale: Literal["days", "hours"] = "days",
	agg: Literal["sum", "mean"] = "sum",
	scope: FacetScope = "cages",
	unit: Unit | Literal["auto"] = "auto",
	hours_range: tuple[int, int] | None = None,
	label_by: LabelBy = "animal_id",
) -> go.Figure:
	"""Time spent in each cage, or each tunnel, across the window or the hours of the day.

	``"days"`` steps through the window's days or phases, following ``granularity``;
	``"hours"`` folds the window onto the 24 hours of the experiment day.
	"""
	window = _window(context, days_range, granularity)
	x = granularity if timescale == "days" else "hour"
	heatmap = prepare.prep_time_per_position(
		context,
		window,
		agg,
		granularity,
		x,
		context.scope_positions(scope),
		context.animal_ids,
		unit,
		hours_range,
	)
	heatmap = replace(heatmap, y=animal_labels(context, label_by))

	spans = prepare.prep_event_spans(context, window, granularity, x, hours_range)
	kind = "daily" if timescale == "days" else "hourly"

	return plot_factory.plot_time_spent_per_cage(
		heatmap, kind, spans, SCOPE_NOUN[scope], granularity
	)


@PlotRegistry.register(
	"activity-line",
	title="Activity over time",
	info=(
		"Antenna reads per animal, counted per hour of each day or phase, then folded onto the "
		"24 hours of the day or onto the window's days/phases. Sum draws the totals; mean draws "
		"the average with a shaded standard-error band. The side panel shows each line's total "
		"or spread."
	),
	requires=("main_df", "animals"),
	dynamic_choices=BY_COHORT,
)
def activity_line(
	context: PlotContext,
	*,
	days_range: tuple[int, int] | None = None,
	granularity: Granularity = "day",
	timescale: Literal["days", "hours"] = "hours",
	agg: Literal["sum", "mean"] = "sum",
	color_by: str = "animal_id",
	label_by: LabelBy = "animal_id",
	hours_range: tuple[int, int] | None = None,
	group_mean: bool = False,
) -> go.Figure:
	"""Antenna detections per hour of the day, or per day of the window.

	``"hours"`` folds the window onto the 24 hours of the experiment day, showing the
	circadian rhythm; ``"days"`` steps through the window's days or phases, following
	``granularity``. For the mean, a shaded band shows the standard error across the axis
	folded away.
	"""
	window = _window(context, days_range, granularity)
	x = granularity if timescale == "days" else "hour"
	frame = prepare.prep_count_line(
		context, window, granularity, x, "main_df", "animal_id", pl.len(), hours_range
	)
	spans = prepare.prep_event_spans(context, window, granularity, x, hours_range)
	mapping = resolve_colors(context, color_by, group_mean=group_mean, label_by=label_by)
	frame = mean_by_group(frame, mapping, ["total", "mean"])

	return LINE_BY_AGG[agg](frame, mapping, "activity", x, context.phases, spans)


@PlotRegistry.register(
	"chasings-line",
	title="Chasings over time",
	info=(
		"Chasings per animal, as chaser or chased: one animal following another through the "
		"same tunnel within a set time window. Counts per hour are folded onto the hours of the "
		"day or onto days/phases; mean adds a shaded standard-error band."
	),
	requires=("chasings_df", "animals"),
	dynamic_choices=BY_COHORT,
)
def chasings_line(
	context: PlotContext,
	*,
	days_range: tuple[int, int] | None = None,
	granularity: Granularity = "day",
	timescale: Literal["days", "hours"] = "hours",
	agg: Literal["sum", "mean"] = "sum",
	scope: Literal["chaser", "chased"] = "chaser",
	color_by: str = "animal_id",
	label_by: LabelBy = "animal_id",
	hours_range: tuple[int, int] | None = None,
	group_mean: bool = False,
) -> go.Figure:
	"""Chasings per hour of the day, or per day of the window.

	Counts each animal's chasings as the chaser, or as the chased. ``timescale`` works as
	for ``activity-line``. For the mean, a shaded band shows the standard error across the
	axis folded away.
	"""
	window = _window(context, days_range, granularity)
	x = granularity if timescale == "days" else "hour"
	frame = prepare.prep_count_line(
		context, window, granularity, x, "chasings_df", scope, pl.sum("chasings"), hours_range
	)
	spans = prepare.prep_event_spans(context, window, granularity, x, hours_range)
	mapping = resolve_colors(
		context, color_by, animal_column=scope, group_mean=group_mean, label_by=label_by
	)
	frame = mean_by_group(frame, mapping, ["total", "mean"])

	return LINE_BY_AGG[agg](frame, mapping, "chasings", x, context.phases, spans)


@PlotRegistry.register(
	"ranking-line",
	title="Dominance ranking",
	info=(
		"Every chasing is replayed in time order as a match the chaser won, updating each "
		"animal's Plackett-Luce rating. Intime draws each animal's ordinal rating (mu - 3 "
		"sigma) after every match; stability ranks animals by their last rating in each day or "
		"phase, rank 1 on top."
	),
	requires=("ranking", "animals"),
	dynamic_choices=BY_COHORT,
)
def ranking_over_time(
	context: PlotContext,
	*,
	days_range: tuple[int, int] | None = None,
	granularity: Granularity = "day",
	mode: Literal["intime", "stability"] = "intime",
	color_by: str = "animal_id",
	label_by: LabelBy = "animal_id",
) -> go.Figure:
	"""Ranking over time, or its day-to-day stability."""
	window = _window(context, days_range, granularity)
	mapping = resolve_colors(context, color_by, label_by=label_by)

	match mode:
		case "intime":
			frame = prepare.prep_ranking_over_time(context, window, granularity)
			spans = prepare.prep_event_spans(context, window, granularity, "datetime")

			return plot_factory.plot_ranking_line(frame, mapping, spans)
		case "stability":
			frame = prepare.prep_ranking_day_stability(context, window, granularity)
			spans = prepare.prep_event_spans(context, window, granularity, granularity)

			return plot_factory.plot_ranking_stability(frame, mapping, granularity, spans)


@PlotRegistry.register(
	"ranking-distribution-line",
	title="Ranking distribution",
	info=(
		"Each animal's rating after the last chasing in the window is a normal distribution: mu "
		"is its estimated skill, sigma the uncertainty. The curve is that density, so its peak "
		"shows where the animal ranks and its width how sure the model is; overlapping curves "
		"mean an unsettled order."
	),
	requires=("ranking", "animals"),
	dynamic_choices=BY_COHORT,
)
def ranking_distribution(
	context: PlotContext,
	*,
	days_range: tuple[int, int] | None = None,
	granularity: Granularity = "day",
	color_by: str = "animal_id",
	label_by: LabelBy = "animal_id",
) -> go.Figure:
	"""Probability density of each animal's ranking after the last match in range."""
	window = _window(context, days_range, granularity)
	frame = prepare.prep_ranking_distribution(context, window, granularity)
	mapping = resolve_colors(context, color_by, label_by=label_by)

	return plot_factory.plot_ranking_distribution(frame, mapping)


@PlotRegistry.register(
	"metrics-polar-line",
	title="Feature overview",
	info=(
		"Per-animal rates of activity, time alone, time together, encounters and chasings, each "
		"normalised by its exposure (hours observed, or per partner). Each metric is z-scored "
		"across animals and phases within the selection, then averaged per animal; the shaded "
		"band is the standard error across days or phases."
	),
	requires=("feature_df", "animals"),
	dynamic_choices=BY_COHORT_AND_PHASE,
)
def polar_metrics(
	context: PlotContext,
	*,
	days_range: tuple[int, int] | None = None,
	granularity: Granularity = "day",
	phase_type: Sequence[str] = PHASES,
	color_by: str = "animal_id",
	label_by: LabelBy = "animal_id",
	hours_range: tuple[int, int] | None = None,
	group_mean: bool = False,
) -> go.Figure:
	"""Z-scored dominance, activity and proximity metrics on one polar scale.

	A polar overlay of metrics in different units is unreadable raw, so each is
	z-scored for display only.
	"""
	window = _window(context, days_range, granularity)
	frame = prepare.prep_polar(context, window, phase_type, granularity, hours_range)
	mapping = resolve_colors(context, color_by, group_mean=group_mean, label_by=label_by)
	frame = mean_by_group(frame, mapping, ["mean"])

	return plot_factory.plot_metrics_polar(frame, mapping)


@PlotRegistry.register(
	"chasings-heatmap",
	title="Chasings matrix",
	info=(
		"Number of chasings for every chaser (column) and chased (row) pair, over the selected "
		"phases and window. Sum totals them; mean averages the per-day (or per-phase) counts. "
		"Brighter cells mean more chasings; compare a cell with its mirror across the diagonal "
		"to see who dominates the pair."
	),
	requires=("chasings_df", "animals"),
	dynamic_choices=PHASE_TYPE,
)
def chasings_heatmap(
	context: PlotContext,
	*,
	days_range: tuple[int, int] | None = None,
	granularity: Granularity = "day",
	phase_type: Sequence[str] = PHASES,
	agg: Literal["sum", "mean"] = "sum",
	hours_range: tuple[int, int] | None = None,
	label_by: LabelBy = "animal_id",
) -> go.Figure:
	"""Chaser-versus-chased matrix of agonistic interactions.

	Columns are chasers and rows are chased.
	"""
	window = _window(context, days_range, granularity)
	animals = context.animal_ids
	matrix = prepare.prep_directed_heatmap(
		context,
		window,
		phase_type,
		agg,
		granularity,
		animals,
		table="chasings_df",
		value="chasings",
		column="chaser",
		row="chased",
		hours_range=hours_range,
	)

	return plot_factory.plot_heatmap(
		matrix,
		animal_labels(context, label_by),
		"<b>Chasings</b>",
		("Chaser", "Chased", "Number"),
	)


@PlotRegistry.register(
	"tube-test-heatmap",
	title="Tube-test matrix",
	info=(
		"Spontaneous tube tests: two animals enter one tunnel from opposite ends and one backs "
		"out to the cage it came from. Cells count wins for every winner (column) and loser "
		"(row) pair; sum totals them, mean averages per day or phase. Compare mirrored cells to "
		"see who dominates."
	),
	requires=("tube_test_df", "animals"),
	dynamic_choices=PHASE_TYPE,
)
def tube_test_heatmap(
	context: PlotContext,
	*,
	days_range: tuple[int, int] | None = None,
	granularity: Granularity = "day",
	phase_type: Sequence[str] = PHASES,
	agg: Literal["sum", "mean"] = "sum",
	hours_range: tuple[int, int] | None = None,
	label_by: LabelBy = "animal_id",
) -> go.Figure:
	"""Winner-versus-loser matrix of spontaneous tube-test outcomes."""
	window = _window(context, days_range, granularity)
	animals = context.animal_ids
	matrix = prepare.prep_directed_heatmap(
		context,
		window,
		phase_type,
		agg,
		granularity,
		animals,
		table="tube_test_df",
		value="tube_test",
		column="winner",
		row="loser",
		hours_range=hours_range,
	)

	return plot_factory.plot_heatmap(
		matrix,
		animal_labels(context, label_by),
		"<b>Spontaneous tube-test</b>",
		("Winner", "Loser", "Number"),
	)


@PlotRegistry.register(
	"sociability-heatmap",
	title="Pairwise sociability",
	info=(
		"Time each pair spent together at a position, or how many separate meetings they had, "
		"one matrix per cage or tunnel. Co-presence shorter than the minimum meeting time is "
		"dropped. Sum totals the window; mean averages the hourly values."
	),
	requires=("pairwise_meetings", "animals"),
	dynamic_choices=PHASE_TYPE,
)
def pairwise_sociability(
	context: PlotContext,
	*,
	days_range: tuple[int, int] | None = None,
	granularity: Granularity = "day",
	phase_type: Sequence[str] = PHASES,
	agg: Literal["sum", "mean"] = "sum",
	metric: Literal["time_together", "pairwise_encounters"] = "time_together",
	scope: FacetScope = "cages",
	unit: Unit | Literal["auto"] = "auto",
	hours_range: tuple[int, int] | None = None,
	label_by: LabelBy = "animal_id",
) -> go.Figure:
	"""How often pairs meet, or how long they spend together, per cage or per tunnel."""
	window = _window(context, days_range, granularity)
	heatmap = prepare.prep_pairwise_sociability(
		context,
		window,
		phase_type,
		agg,
		metric,
		granularity,
		context.scope_positions(scope),
		context.animal_ids,
		unit,
		hours_range,
	)

	labels = animal_labels(context, label_by)

	return plot_factory.plot_sociability_heatmap(replace(heatmap, x=labels, y=labels), metric)


@PlotRegistry.register(
	"cohort-heatmap",
	title="Within-cohort sociability",
	info=(
		"Proportion together is the share of each phase a pair spent together at the selected "
		"positions. Sociability is the cage share minus the chance expectation from each "
		"animal's own cage time (tA·tB/T²), summed over cages; positive means the pair sought "
		"each other out. Cells average the selected phases."
	),
	requires=("incohort_sociability", "pairwise_meetings", "phase_durations", "animals"),
	dynamic_choices=PHASE_TYPE,
)
def within_cohort_sociability(
	context: PlotContext,
	*,
	days_range: tuple[int, int] | None = None,
	granularity: Granularity = "day",
	phase_type: Sequence[str] = PHASES,
	metric: Literal["proportion_together", "sociability"] = "proportion_together",
	scope: Scope = "all",
	label_by: LabelBy = "animal_id",
) -> go.Figure:
	"""Mean sociability index between every pair in the cohort.

	``scope`` applies to ``proportion_together`` only: ``sociability`` is measured against
	a cage-only chance expectation, so it ignores the selection.
	"""
	window = _window(context, days_range, granularity)
	animals = context.animal_ids
	matrix = prepare.prep_within_cohort_sociability(
		context, window, phase_type, metric, granularity, animals, scope
	)

	title = (
		"<b>Proportional time spent together</b>"
		if metric == "proportion_together"
		else "<b>Within-cohort sociability</b>"
	)

	return plot_factory.plot_heatmap(
		matrix, animal_labels(context, label_by), title, ("X", "Y", "Sociability")
	)


@PlotRegistry.register(
	"social-stability",
	title="Relationship stability",
	info=(
		"Each animal gets one point per partner, coloured by that animal: height is the median "
		"share of each day (or phase) the pair spent together. Stability, along x, is 1 - "
		"MAD/median of those shares, clipped to 0-1; points further right are steadier "
		"relationships."
	),
	requires=("pairwise_meetings", "phase_durations", "animals"),
	dynamic_choices=BY_COHORT_AND_PHASE,
)
def social_stability(
	context: PlotContext,
	*,
	days_range: tuple[int, int] | None = None,
	granularity: Granularity = "day",
	phase_type: Sequence[str] = PHASES,
	scope: Scope = "all",
	color_by: str = "animal_id",
	label_by: LabelBy = "animal_id",
) -> go.Figure:
	"""Stability of every pair's relationship against how much time they share."""
	window = _window(context, days_range, granularity)
	frame = prepare.prep_social_stability(context, window, phase_type, granularity, scope)
	mapping = resolve_colors(context, color_by, label_by=label_by)

	return plot_factory.plot_social_stability(frame, mapping)


@PlotRegistry.register(
	"quality-heatmap",
	title="Missed passes by animal and antenna",
	info=(
		"A missed pass is one the antenna did not read: interpolated by preprocessing, or "
		"still unresolved (bad). Cells show the share of each animal's passes per antenna "
		"that were missed, a lower bound, inside the analysed window. Bright columns flag weak "
		"antennas; bright rows, weak tags."
	),
	requires=("recording_quality", "animals"),
)
def quality_heatmap(context: PlotContext, *, label_by: LabelBy = "animal_id") -> go.Figure:
	"""Share of each animal's passes over each antenna that went unrecorded."""
	matrix, antennas = prepare.prep_quality_heatmap(context, context.animal_ids)

	return plot_factory.plot_quality_heatmap(matrix, animal_labels(context, label_by), antennas)


@PlotRegistry.register(
	"quality-antenna",
	title="Reads per antenna",
	info=(
		"Every pass over each antenna in the analysed window. Correct: read. Interpolated: "
		"inserted by preprocessing. Bad: the layout says the animal crossed it unread and "
		"nothing filled it in. Steps across a recording stop don't count. Bars pool the "
		"cohort; a thick top flags a marginal antenna."
	),
	requires=("recording_quality",),
)
def quality_by_antenna(context: PlotContext) -> go.Figure:
	"""Pooled over the cohort, so a marginal antenna stands out first."""
	return plot_factory.plot_quality_by_antenna(prepare.prep_quality_by_antenna(context))


@PlotRegistry.register(
	"network-dominance",
	title="Dominance network",
	info=(
		"Nodes are animals, sized by their latest dominance rating; arrows run from chaser to "
		"chased, their width and colour scaled by the number of chasings in the window. Edges "
		"under the cutoff (percent of the strongest) are dropped before the layout is fitted; "
		"spring pulls frequent pairs together."
	),
	requires=("chasings_df", "ranking", "animals"),
	dynamic_choices=BY_COHORT,
)
def network_dominance(
	context: PlotContext,
	*,
	days_range: tuple[int, int] | None = None,
	granularity: Granularity = "day",
	layout: Literal["spring", "circular"] = "spring",
	edge_cutoff: float = 0,
	color_by: str = "animal_id",
	label_by: LabelBy = "animal_id",
) -> go.Figure:
	"""Directed network of chasing, with node size showing ranking.

	``edge_cutoff`` drops edges weaker than that percentage of the strongest before the
	layout is fitted, so the network is recomputed from what remains.
	"""
	window = _window(context, days_range, granularity)
	connections, nodes = prepare.prep_network_dominance(context, window, granularity)
	mapping = resolve_colors(context, color_by, label_by=label_by)
	colors = [mapping.by_animal[animal] for animal in context.animal_ids]

	figure = plot_factory.plot_network_graph(
		connections,
		nodes,
		context.animal_ids,
		colors,
		"chasings",
		layout,
		edge_cutoff,
		animal_labels(context, label_by),
	)
	return figure.update_layout(colorway=list(mapping.colors.values()))


@PlotRegistry.register(
	"network-sociability",
	title="Sociability network",
	info=(
		"Nodes are animals; each edge is the pair's share of each phase spent together at the "
		"selected positions, summed over the window, its width and colour scaled to the "
		"cohort's range. Edges under the cutoff (percent of the strongest) are dropped before "
		"the layout is fitted."
	),
	requires=("pairwise_meetings", "phase_durations", "animals"),
	dynamic_choices=BY_COHORT,
)
def network_sociability(
	context: PlotContext,
	*,
	days_range: tuple[int, int] | None = None,
	granularity: Granularity = "day",
	layout: Literal["spring", "circular"] = "spring",
	edge_cutoff: float = 0,
	scope: Scope = "all",
	color_by: str = "animal_id",
	label_by: LabelBy = "animal_id",
) -> go.Figure:
	"""Undirected network weighted by the time each pair spends together.

	``edge_cutoff`` works as for ``network-dominance``.
	"""
	window = _window(context, days_range, granularity)
	connections = prepare.prep_network_sociability(context, window, granularity, scope)
	mapping = resolve_colors(context, color_by, label_by=label_by)
	colors = [mapping.by_animal[animal] for animal in context.animal_ids]

	figure = plot_factory.plot_network_graph(
		connections,
		None,
		context.animal_ids,
		colors,
		"proportion_together",
		layout,
		edge_cutoff,
		animal_labels(context, label_by),
	)
	return figure.update_layout(colorway=list(mapping.colors.values()))


@PlotRegistry.register(
	"recording-pulse",
	title="Recording pulse",
	info=(
		"Cohort visits summed per hour, one row per experiment day, so gaps and rhythm drifts "
		"stand out. Blank cells are hours with nothing recorded; the vertical line marks the "
		"phase switch and outlined cells an event."
	),
	requires=("activity_df",),
)
def recording_pulse(
	context: PlotContext,
	*,
	days_range: tuple[int, int] | None = None,
	granularity: Granularity = "day",
	hours_range: tuple[int, int] | None = None,
) -> go.Figure:
	"""Cohort visits per hour, one row per experiment day, outlined where an event ran.

	Rows stay experiment days whichever granularity the window is in, since a phase
	covers half a day: a phase window narrows which hours of those days are drawn.
	"""
	window = _window(context, days_range, granularity)
	heatmap, cells = prepare.prep_actogram(context, window, granularity, hours_range)

	return plot_factory.plot_actogram(heatmap, cells, context.phases)


@PlotRegistry.register(
	"habitat-occupancy",
	title="Habitat occupancy",
	info=(
		"Share of the cohort's total animal-time held by each cage, all tunnels together, and "
		"undefined, per day or phase, stacked to 100%. Undefined (hatched) is time no antenna "
		"could place. A band that thins over days shows the cohort abandoning that cage."
	),
	requires=("activity_df",),
)
def habitat_occupancy(
	context: PlotContext,
	*,
	days_range: tuple[int, int] | None = None,
	granularity: Granularity = "day",
	hours_range: tuple[int, int] | None = None,
) -> go.Figure:
	"""Share of the cohort's time each place held, across the window.

	Position preference collapsed over time answers where the group lived; this keeps
	the time axis, so a cohort abandoning a cage shows up as the move it was.
	"""
	window = _window(context, days_range, granularity)
	frame = prepare.prep_occupancy_share(context, window, granularity, hours_range)
	spans = prepare.prep_event_spans(context, window, granularity, granularity, hours_range)

	# Cages take the palette cage-preference gives them at cages-only scope, so a cage
	# keeps one colour across the dashboard. Tunnels and undefined are not cages and stay
	# out of the hue budget: one neutral band for transit, a hatched one for no place.
	colors = dict(zip(context.cages, sample_palette(len(context.cages)), strict=True))
	colors |= {"tunnels": "rgba(123, 137, 136, 0.5)", Layout.UNDEFINED: "rgba(123, 137, 136, 0.22)"}
	order = [*context.cages, "tunnels", Layout.UNDEFINED]

	return plot_factory.plot_occupancy_ribbon(frame, order, colors, granularity, spans)


@PlotRegistry.register(
	"cohort-phenotype",
	title="Cohort phenotype map",
	info=(
		"One marker per animal: x is visits (locomotion), y the share of its cage time spent "
		"with company (1 - time alone / cage time). Marker area is the dominance rating, "
		"measured from the lowest; hover adds chases won, and the top-ranked animal is "
		"annotated against the cohort median."
	),
	requires=("activity_df", "chasings_df", "ranking", "animals"),
	dynamic_choices=BY_COHORT,
)
def cohort_phenotype(
	context: PlotContext,
	*,
	days_range: tuple[int, int] | None = None,
	granularity: Granularity = "day",
	hours_range: tuple[int, int] | None = None,
	color_by: str = "animal_id",
	label_by: LabelBy = "animal_id",
) -> go.Figure:
	"""One marker per animal, carrying locomotion, sociality, chasing and rank at once.

	Marker area is the dominance rating, so an animal that sits apart from the cloud is
	the one to open the Social or Dominance tab for.
	"""
	window = _window(context, days_range, granularity)
	mapping = resolve_colors(context, color_by, label_by=label_by)
	return plot_factory.plot_phenotype_map(
		prepare.prep_phenotype(context, window, granularity, hours_range), mapping, label_by
	)
