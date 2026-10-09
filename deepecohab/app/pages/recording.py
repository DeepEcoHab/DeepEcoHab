import datetime as dt
import json
import math
from itertools import pairwise
from urllib.parse import parse_qs, quote, urlencode

import dash
import dash_mantine_components as dmc
import plotly.graph_objects as go
from dash import (
	ALL,
	MATCH,
	ClientsideFunction,
	Input,
	Output,
	State,
	callback,
	clientside_callback,
	ctx,
	dcc,
	html,
	no_update,
)
from dash.exceptions import PreventUpdate
from plotly.colors import qualitative

from deepecohab.app import components, services
from deepecohab.app.components import icon, notify
from deepecohab.core import topology
from deepecohab.core.data_model import (
	LEAD_WARNING_THRESHOLD,
	DataFrameRegistry,
	Recording,
	Timeline,
)
from deepecohab.plotting import PlotContext, PlotRegistry, available_attributes, theme as plot_theme
from deepecohab.plotting.animals import resolve_colors
from deepecohab.plotting.plot_catalog import PHASES
from deepecohab.plotting.plot_factory import FACETS_PER_LINE
from deepecohab.plotting.prepare import READ_KINDS
from deepecohab.plotting.theme import COLORSCALES, PALETTES

PATH = "/recording"

dash.register_page(
	__name__, path=PATH, name="Recording dashboard", order=1, icon="layout-dashboard"
)

#: Driven by the control bar rather than a per-card control: builder option -> rec-controls key.
_GLOBAL_OPTIONS = {
	"days_range": "window",
	"granularity": "granularity",
	"phase_type": "phases",
	"color_by": "color_by",
	"label_by": "label_by",
	"hours_range": "hours",
	"group_mean": "group_mean",
}
#: The filters a card can ignore, as (icon, label, tooltip); deh.plotRequest shows the badge
#: while its control is narrowed.
_BADGES = {
	"window": (
		"calendar",
		"Whole recording",
		"This card always covers the whole recording; the window does not apply to it.",
	),
	"hours": (
		"clock",
		"Whole day",
		"This card has no hourly breakdown; the hours window does not apply to it.",
	),
	"phases": (
		"moon",
		"Both phases",
		"This card does not split by phase; the phase filter does not apply to it.",
	),
	"group_mean": (
		"users",
		"Per animal",
		"Group mean does not apply here; this card shows one line per animal.",
	),
}
_OPTION_LABELS = {"agg": "Aggregate", "edge_cutoff": "Hide edges below"}
#: A control the recording has no use for is hidden, not dropped: its callbacks take it as an
#: Input, and Dash throws on a callback whose inputs are only partly in the layout.
_HIDDEN = {"display": "none"}
#: Height of the Position-unknown table, matching the heatmap it shares a grid row with.
_MISSING_TABLE_H = 430
#: (id, label, cells); a cell is (plot, column span of 12, height px[, grid rows]).
#: "cohort", "events", "overview-summary" and "quality-summary" are computed cards, not
#: registered plots; "events" is left out of a recording that declares none. 548px is a
#: half-width (span 6) card's rendered width at the common 1440px desktop viewport this app
#: is designed around, so a plot with that height there reads square.
#: Diagnostics comes first, and is what a recording opens on: whether the acquisition can be
#: trusted is the question to settle before reading anything the analysis says.
_SECTIONS = [
	(
		"diagnostics",
		"Diagnostics",
		[
			("quality-summary", 12, 0),
			("habitat", 5, 500),
			("quality-antenna", 7, 500),
			("quality-heatmap", 5, 430),
			("quality-missing", 7, _MISSING_TABLE_H),
		],
	),
	(
		"overview",
		"Overview",
		[
			("overview-summary", 12, 0),
			("metrics-polar-line", 6, 500),
			("cohort", 6, 500),
			("recording-pulse", 12, 380),
			("habitat-occupancy", 7, 360),
			("cohort-phenotype", 5, 360),
			("events", 12, 0),
		],
	),
	(
		"activity",
		"Activity",
		[
			("recording-timeline", 12, 340),
			("activity-line", 8, 360),
			("cage-preference", 4, 360),
			("activity-bar", 12, 320),
			("animal-speed", 6, 380),
			("animal-speed-daily", 6, 380),
			("cage-preference-evolution", 12, 660),
		],
	),
	(
		"social",
		"Social",
		[
			("sociability-heatmap", 12, 340),
			("cohort-heatmap", 5, 380),
			("time-alone-bar", 7, 380),
			("network-sociability", 6, 424),
			("social-stability", 6, 424),
		],
	),
	(
		"dominance",
		"Dominance",
		[
			("ranking-line", 12, 360),
			("chasings-line", 8, 360),
			("ranking-distribution-line", 4, 360),
			("network-dominance", 6, 424),
			("chasings-heatmap", 6, 424),
		],
	),
]
_TAB_IDS = {section_id for section_id, *_ in _SECTIONS}


layout = html.Div(
	[
		dcc.Store(id="rec-context"),
		dcc.Store(id="rec-controls"),
		# rec-tabs is built into rec-body, so a callback that also takes an input from this
		# static layout would be dispatched before it exists ("a nonexistent object was used
		# in an Input"). switchTab mirrors the tab here, where plotRequest can always see it.
		dcc.Store(id="rec-tab"),
		# The tab the pointer rests on, as {tab, at}, for plotRequest to prefetch.
		dcc.Store(id="rec-hover"),
		# What the cohort cards were drawn for: _dashboard bakes them for animal_id.
		dcc.Store(id="rec-color-by", data="animal_id"),
		# The timeline's zoomed x range, or null for its whole window.
		dcc.Store(id="rec-timeline-zoom"),
		dcc.Store(id="notes-target"),
		dcc.Store(id="rec-format", storage_type="session"),
		dcc.Store(id="rec-format-target"),
		dcc.Store(id="rec-colors", data={"colorscale": COLORSCALES, "palette": PALETTES}),
		components.format_dialog(
			"rec", "Empty means automatic. Applies to this plot on every recording."
		),
		html.Div(id="rec-body"),
		dmc.Modal(
			id="fullscreen-modal",
			title="",
			fullScreen=True,
			classNames=components.DIALOG_CLASSES,
			children=html.Div(
				dcc.Graph(
					id="fullscreen-plot",
					figure=components.EMPTY_FIGURE,
					config={"displayModeBar": False},
				),
				id="fullscreen-frame",
			),
		),
		dmc.Modal(
			id="notes-modal",
			title="Notes",
			size=520,
			classNames=components.DIALOG_CLASSES,
			children=[
				html.Div(
					dmc.Textarea(
						id="notes-textarea",
						placeholder="No notes",
						autosize=True,
						minRows=6,
						className="deh-field",
					),
					className="deh-dialog-body",
				),
				html.Div(
					[
						html.Button("Cancel", id="notes-cancel", className="deh-btn deh-btn-ghost"),
						html.Button("Save", id="notes-save", className="deh-btn deh-btn-primary"),
					],
					className="deh-dialog-foot",
				),
			],
		),
	]
)


def _human(value: str) -> str:
	return str(value).replace("_", " ")


def _reads(tables: tuple[str, ...]) -> list:
	children = ["reads "]
	for index, table in enumerate(tables):
		if index:
			children.append(" ")
		children.append(html.Code(table))
	return children


def _card_head(title: str, blurb, *, maximize: bool = False) -> html.Div:
	# The card itself fills the window (deh-card-max in clientside.js) rather than a copy in a
	# modal, so its table rows and map stay the live, clickable ones.
	button = html.Button(
		[icon("maximize", size=15), icon("x", size=15)],
		className="deh-icon-btn sm deh-card-max",
		title=f"Open {title} full screen",
	)
	return html.Div(
		[
			html.Div([html.H3(title), html.P(blurb)], className="deh-card-titles"),
			html.Div(button, className="deh-card-actions") if maximize else None,
		],
		className="deh-card-head",
	)


def _needs(*tables: str, hint: str | None = None) -> html.Div:
	return html.Div(
		[
			icon("database", size=22),
			html.Div([html.B(["Needs ", *_reads(tables)[1:]]), hint and html.P(hint)]),
		],
		className="deh-missing-body",
	)


def _quality_badge(miss: float) -> html.Span:
	if miss < 1:
		kind, glyph, label = "good", "circle-check", "Good"
	elif miss < 2.5:
		kind, glyph, label = "warn", "alert-triangle", "Check"
	else:
		kind, glyph, label = "bad", "circle-x", "Poor"
	return html.Span([icon(glyph, size=14), label], className=f"deh-badge deh-badge-{kind}")


def _meta_strip(summary: dict) -> list:
	start = dt.datetime.fromisoformat(summary["start"])
	end = dt.datetime.fromisoformat(summary["end"])
	quality = summary["quality"]
	# Grouped by what they answer (when / where and who / what to follow up), so a narrow
	# header wraps between groups rather than through one.
	groups = [
		[
			html.Span(
				[icon("calendar", size=15), f"{start.day} {start:%b} → {end.day} {end:%b %Y}"],
				title=(
					f"Start: {start:%a} {start.day} {start:%b %Y, %H:%M:%S}\n"
					f"End: {end:%a} {end.day} {end:%b %Y, %H:%M:%S}\n"
					f"Time zone: {summary['timezone']}"
				),
			),
			html.Span(
				[
					icon("clock", size=15),
					f"{summary['days']} days · {summary['phases']} phases",
				],
				title="\n".join(
					[
						f"Starts with the {_human(summary['start_from'])}",
						*(
							f"{_human(phase).capitalize()} onset: {at}"
							for phase, at in summary["onsets"].items()
						),
					]
				),
			),
		],
		[
			*(
				[
					html.Span(
						[icon("map-pin", size=15), summary["location"]], title="Recording location"
					)
				]
				if summary["location"]
				else []
			),
			html.Button(
				[icon("users", size=15), f"{summary['n_mice']} mice"],
				id="rec-mice-jump",
				className="deh-btn deh-btn-ghost sm",
				title="Show the cohort",
			),
			html.Button(
				[
					icon("grid-dots", size=15),
					f"{summary['cages']} cages · {summary['tunnels']} tunnels",
				],
				id="rec-habitat-jump",
				className="deh-btn deh-btn-ghost sm",
				title="Show the habitat map",
			),
		],
		[
			html.Button(
				[icon("bolt", size=15), f"{len(summary['events'])} events"],
				id="rec-events-jump",
				className="deh-btn deh-btn-ghost sm",
				title=", ".join(summary["events"]) or "No events",
				disabled=not summary["events"],
			),
			*(
				[
					html.Button(
						[f"{quality['miss']:.2f}% missed ", _quality_badge(quality["miss"])],
						id="rec-quality-jump",
						className="deh-btn deh-btn-ghost sm",
					)
				]
				if quality is not None
				else []
			),
			html.Button(
				[icon("notes", size=15), "Notes"],
				id="rec-notes-btn",
				className="deh-btn deh-btn-ghost sm",
			),
		],
	]
	return [html.Div(group, className="deh-meta-group") for group in groups]


def _shade(phase: str, selected: bool) -> str:
	"""One phase's colour for a band, dimmed when the chips have turned it off."""
	token = "--tick-dark" if phase == "dark_phase" else "--tick-light"
	return f"color-mix(in srgb, var({token}) {100 if selected else 22}%, transparent)"


def _cohort_widgets(context: PlotContext, color_by: str) -> tuple[list, html.Div]:
	"""The cohort popover trigger's children and the popover body, for one colour attribute."""
	if "animals" not in context:
		empty = html.Div(html.P("Needs the animals table.", className="deh-sub"))
		return [f"Cohort · {len(context.animal_ids)}"], empty

	mapping, animals = resolve_colors(context, color_by), context.animals.sort("animal_id")
	strip = html.Span(
		[html.I(style={"background": mapping.by_animal[tag]}) for tag in context.animal_ids],
		className="deh-mini-strip",
	)
	rows = [
		html.Li(
			[
				html.Span(
					className="deh-dot", style={"background": mapping.by_animal[row["animal_id"]]}
				),
				html.Span(
					[
						html.Span(row["animal_id"], className="deh-mono"),
						html.Span(
							f" · {row['subject_name']} · {row['sex']} · {row['genotype']} · "
							f"{row['age'].days} d",
							className="deh-sub",
						),
					]
				),
			],
			id={"type": "animal-row", "scope": "popover", "tag": row["animal_id"]},
			n_clicks=0,
			style={"cursor": "pointer"},
			title="Edit notes",
		)
		for row in animals.iter_rows(named=True)
	]
	pop = html.Div(
		[
			html.P(["Colour follows ", html.B(_human(color_by)), "."], className="deh-sub"),
			html.Ul(rows, className="deh-cohort-list"),
		]
	)
	return [strip, f"Cohort · {len(context.animal_ids)}"], pop


def _option_control(plot: str, option) -> html.Div:
	control_id = {"type": "card-opt", "plot": plot, "option": option.name}
	label = _OPTION_LABELS.get(option.name, option.label)
	if not option.choices:
		is_dwell = option.name == "max_dwell"
		control = dmc.NumberInput(
			id=control_id,
			value=option.default,
			min=0.001 if is_dwell else 0,
			max=None if is_dwell else 100,
			step=1 if is_dwell else 5,
			suffix=" s" if is_dwell else "%",
			clampBehavior="strict",
			debounce=400,
			size="xs",
			w=80,
		)
		return html.Div([html.Span(label, className="deh-opt-label"), control], className="deh-opt")

	data = [{"value": str(choice), "label": _human(choice)} for choice in option.choices]
	control = (
		dmc.SegmentedControl(id=control_id, data=data, value=str(option.default), size="xs")
		if len(option.choices) <= 3
		else dmc.Select(
			id=control_id,
			data=data,
			value=str(option.default),
			size="xs",
			w=150,
			allowDeselect=False,
		)
	)
	return html.Div([html.Span(label, className="deh-opt-label"), control], className="deh-opt")


def _card_frame(
	children: list, span: int, rows: int, *, card_id: str | dict | None = None
) -> html.Article:
	attrs: dict[str, object] = {"data-span": str(span), "data-rows": str(rows)}
	if card_id is not None:
		attrs["id"] = card_id
	return html.Article(children, className="deh-card", **attrs)  # ty: ignore[invalid-argument-type]


def _plot_card(name: str, context: PlotContext, height: int, tab: str) -> list:
	spec = PlotRegistry.spec(name)
	missing = [table for table in spec.requires if table not in context]
	accepted = {option.name for option in spec.options}
	uses = [key for option, key in _GLOBAL_OPTIONS.items() if option in accepted]
	badges = [
		html.Span(
			[icon(glyph, size=14), label],
			id={"type": "badge", "plot": name, "control": key},
			className="deh-badge deh-badge-neutral",
			title=tooltip,
			hidden=True,
		)
		for key, (glyph, label, tooltip) in _BADGES.items()
		# Group mean only means something to a card that colours by animal at all.
		if key not in uses
		and (key != "group_mean" or "color_by" in uses)
		# The phase chips set the hours window, so an hour-aware card follows them too.
		and (key != "phases" or "hours" not in uses)
	]
	header = html.Div(
		[
			html.Div(
				[
					html.H3(spec.title, id={"type": "card-title", "plot": name}),
					html.P(spec.summary),
				],
				className="deh-card-titles",
			),
			html.Span(None if missing else badges, className="deh-card-badge"),
			html.Div(
				[
					dmc.Menu(
						[
							dmc.MenuTarget(
								html.Button(
									icon("info-circle", size=15),
									className="deh-icon-btn sm",
									title=f"About {spec.title}",
								)
							),
							dmc.MenuDropdown(
								[
									dmc.MenuLabel(spec.title),
									html.P(spec.info, className="deh-menu-text"),
								]
							),
						],
						position="bottom-end",
						classNames={"dropdown": "deh-menu"},
					),
					html.Button(
						icon("format", size=15),
						id={"type": "card-format", "plot": name},
						className="deh-icon-btn sm",
						title=f"Format {spec.title}",
					),
					html.Button(
						icon("download", size=16),
						id={"type": "card-export", "plot": name},
						className="deh-icon-btn sm",
						title=f"Export {spec.title}",
					),
					html.Button(
						icon("maximize", size=15),
						id={"type": "card-fullscreen", "plot": name},
						className="deh-icon-btn sm",
						title=f"Open {spec.title} full screen",
					),
				],
				className="deh-card-actions",
			),
		],
		className="deh-card-head",
	)

	if missing:
		body = _needs(*missing, hint="Run analysis for this recording to build it.")
		return [header, body, html.Footer(_reads(spec.requires), className="deh-card-foot")]

	options = [option for option in spec.options if option.name not in _GLOBAL_OPTIONS]
	return [
		header,
		html.Div([_option_control(name, option) for option in options], className="deh-card-opts")
		if options
		else None,
		dcc.Store(id={"type": "plot-req", "plot": name}, data={"tab": tab, "uses": uses}),
		dcc.Loading(
			dcc.Graph(
				id={"type": "plot", "plot": name},
				figure=components.EMPTY_FIGURE,
				config={"displayModeBar": False, "responsive": True},
				className="deh-plot",
				style={"height": f"{height}px"} if height else {},
			),
			custom_spinner=icon("loader-2", size=28, class_name="deh-spin"),
			delay_show=200,
			parent_className="deh-plot-wrap",
		),
		html.Footer(_reads(spec.requires), className="deh-card-foot"),
	]


def _overview_summary_children(context: PlotContext) -> list:
	header = _card_head(
		"Recording at a glance",
		"What this recording amounts to: how long it ran, how many detections, "
		"and where the cohort spent its time.",
	)
	body = _tiles(
		[
			(tile["label"], tile["value"], html.Span(tile["note"], className="deh-sub"))
			for tile in services.overview_tiles(context)
		]
	)
	reads = [table for table in ("main_df", "activity_df", "chasings_df") if table in context]

	return [header, body, html.Footer(_reads(tuple(reads)), className="deh-card-foot")]


def _quality_summary_children(context: PlotContext) -> list:
	header = _card_head(
		"Detection quality",
		"Passes the antennas should have caught but did not, whether preprocessing "
		"interpolated them or they stay unresolved: an animal that turns up at an antenna "
		"the layout does not join to its previous one crossed antennas that never fired.",
	)
	if "recording_quality" not in context:
		return [header, _needs("recording_quality")]

	assert context.recording is not None
	quality = services.quality_summary(context)
	tiles = [
		("Missed passes", f"{quality['miss']:.2f}%", _quality_badge(quality["miss"])),
		(
			"Correct reads",
			f"{quality['correct']:,}",
			html.Span(
				f"{quality['interpolated']:,} interpolated · {quality['bad']:,} bad",
				className="deh-sub",
			),
		),
	]
	if "activity_df" in context and "phase_durations" in context:
		missing = services.missing_time(context)
		worst = missing["rows"][0]
		tiles.insert(
			1,
			(
				"Position unknown",
				f"{missing['mean_share']:.2f}%",
				html.Span(
					f"Worst: {worst['animal_id']} {worst['share']:.2f}%", className="deh-sub"
				),
			),
		)
	gaps = _gaps(context.recording)
	tiles += [*_window_tiles(context.recording.timeline), _gap_tile(gaps)]
	strip = _gap_strip(context.recording.timeline, gaps) if gaps else None
	return [header, _tiles(tiles), strip]


def _tiles(rows: list) -> html.Div:
	return html.Div(
		[
			html.Div(
				[
					html.Span(label, className="deh-tile-label"),
					html.Span(value, className="deh-tile-value"),
					aside,
				],
				className="deh-tile",
			)
			for label, value, aside in rows
		],
		className="deh-tiles",
	)


def _span(delta: dt.timedelta, rounding=math.ceil) -> str:
	"""A lead or tail in whole minutes, rounded to match the HH:MM clocks shown beside it.

	Clocks drop their seconds, so time before an onset rounds up and time after it down.
	"""
	hours, minutes = divmod(rounding(delta / dt.timedelta(minutes=1)), 60)
	return " ".join(f"{n}{unit}" for n, unit in ((hours, "h"), (minutes, "m")) if n) or "0m"


def _window_note(gap: dt.timedelta, text: str, hover: str) -> html.Span:
	"""A gap off the phase grid, as a warning badge once it is as long as adding one warns of."""
	if gap > LEAD_WARNING_THRESHOLD:
		return html.Span(
			[icon("alert-triangle", size=14), text],
			className="deh-badge deh-badge-warn",
			title=hover,
		)
	return html.Span(text, className="deh-sub", title=hover)


def _window_tiles(line: Timeline) -> list[tuple]:
	"""Acquisition start and end on the clock, each noted with how far it is off the phase grid."""
	onset = f"the {line.phases[line.start_from]:%H:%M} {_human(line.start_from)} onset"
	early, late = line.discarded_lead, line.unrecorded_lead
	cut, tail = line.discarded_tail, line.unrecorded_tail
	closing = f"the end of the {_human(line.end_with)}"
	if early:
		start = _window_note(
			early,
			f"{_span(early)} lost",
			f"Started {_span(early)} before {onset}; what was recorded before it is left out.",
		)
	elif late:
		start = _window_note(
			late,
			f"{_span(late, math.floor)} added",
			f"Started {_span(late, math.floor)} after {onset}, where the grid still opens; "
			"the first phase has no data for that long.",
		)
	else:
		start = _window_note(late, "on the onset", f"Started on {onset}.")
	if cut:
		end = _window_note(
			cut,
			f"{_span(cut)} lost",
			f"Ended {_span(cut)} after {closing}; what was recorded after it is left out.",
		)
	elif tail:
		end = _window_note(
			tail,
			f"{_span(tail)} short",
			f"Ended {_span(tail)} before {closing}; the last phase is short by as much.",
		)
	else:
		end = _window_note(tail, "on an onset", f"Ended on {closing}.")
	zone = line.recording_timezone
	return [
		("Recording start", f"{line.start_datetime.astimezone(zone):%H:%M}", start),
		("Recording end", f"{line.end_datetime.astimezone(zone):%H:%M}", end),
	]


#: What each kind of recording gap means. A clean stop is not a failure.
_GAP_KINDS = {
	"abrupt_end": "Acquisition ended abruptly",
	"observed_record_gap": "Nothing recorded, cause unknown",
	"confirmed_DAQ_inactive": "Acquisition stopped and restarted cleanly",
}


def _gaps(recording: Recording) -> list[tuple[dt.datetime, dt.datetime, str]]:
	"""The recording's gaps as (start, end, kind), clipped to the analysed window."""
	begin, stop = recording.timeline.local_span
	return [
		(max(gap.start, begin), min(gap.end, stop), gap.kind)
		for gap in recording.boundaries
		if gap.end > begin and gap.start < stop
	]


def _gap_tile(gaps: list[tuple[dt.datetime, dt.datetime, str]]) -> tuple:
	"""How often acquisition stopped and for how long.

	Poor past 12 h lost, otherwise warned of unless every stop was clean.
	"""
	if not gaps:
		return ("Recording gaps", "0", html.Span("recorded without a break", className="deh-sub"))
	lost = sum((end - start for start, end, _ in gaps), dt.timedelta())
	failed = sum(kind != "confirmed_DAQ_inactive" for *_, kind in gaps)
	text = f"{_span(lost)} lost"
	note = html.Span(text, className="deh-sub")
	if lost > dt.timedelta(hours=12) or failed:
		kind, glyph = (
			("bad", "circle-x") if lost > dt.timedelta(hours=12) else ("warn", "alert-triangle")
		)
		note = html.Span(
			[icon(glyph, size=14), text],
			className=f"deh-badge deh-badge-{kind}",
			title=f"{failed} of {len(gaps)} were not a clean stop and restart.",
		)
	return ("Recording gaps", str(len(gaps)), note)


def _gap_strip(line: Timeline, gaps: list[tuple[dt.datetime, dt.datetime, str]]) -> html.Div:
	"""Each gap as a bar on the phase-tinted strip the events card draws, coloured by kind."""
	zone = line.recording_timezone
	place, day_phase, band, axis = _strip_scale(line)
	bars = []
	for start, end, kind in gaps:
		meaning = _GAP_KINDS[kind]
		onset, offset = start.astimezone(zone), end.astimezone(zone)
		bars.append(
			html.Span(
				className="deh-ev-bar",
				style={
					**place(onset, offset),
					"background": "var(--bad)",
				},
				title=(
					f"{day_phase(onset)}\n{onset.day} {onset:%b %H:%M} → "
					f"{offset.day} {offset:%b %H:%M} ({_span(end - start)})\n{meaning}"
				),
			)
		)
	lane = [
		html.Span("Gaps", className="deh-ev-label"),
		html.Div(bars, className="deh-ev-track", style={"background": band(False)}),
	]
	return html.Div([*lane, *axis], className="deh-ev-strip")


def _strip_scale(line: Timeline) -> tuple:
	"""A strip across the analysed window, as ``(place, day_phase, band, axis)``.

	``place`` positions a bar from onset to offset, held 3px off the track's rounded ends so
	one at the recording's start or end does not sit flush with them. ``day_phase`` names a
	moment's day and phase, ``band`` tints a lane by phase and ``axis`` is the Day row that
	closes the strip.
	Days and phases are counted the way the window slider counts them: 24 elapsed hours from
	the experiment start.
	"""
	start, end = line.local_span
	origin = start.astimezone(dt.UTC)
	length = end.astimezone(dt.UTC) - origin
	onset_phase = {onset: phase for phase, onset in line.phases.items()}
	stretches = [(start, line.start_from)] + [
		(switch, onset_phase[switch.time()]) for switch in line.phase_boundaries()
	]

	def at(moment: dt.datetime) -> float:
		return 100 * ((moment.astimezone(dt.UTC) - origin) / length)

	def place(onset: dt.datetime, offset: dt.datetime) -> dict:
		return {
			"left": f"max(3px, {at(onset):.3f}%)",
			"right": f"max(3px, {100 - at(offset):.3f}%)",
		}

	def day_phase(moment: dt.datetime) -> str:
		day = (moment.astimezone(dt.UTC) - origin) // dt.timedelta(days=1) + 1
		phase = next(name for switch, name in reversed(stretches) if switch <= moment)
		return f"Day {day} · {_human(phase).removesuffix(' phase')}"

	edges = [at(moment) for moment, _ in stretches] + [100.0]

	def band(selected: bool) -> str:
		stops = ", ".join(
			f"{_shade(phase, selected)} {lo:.3f}% {hi:.3f}%"
			for (_, phase), (lo, hi) in zip(stretches, pairwise(edges), strict=True)
		)
		return f"linear-gradient(90deg, {stops})"

	days = line.days_range[1]
	step = 2 if days > 12 else 1
	day_edges = [min(at(origin + dt.timedelta(days=d)), 100.0) for d in range(days + 1)]
	ticks = [
		html.Span(str(d), className="deh-ev-day", style={"left": f"{(lo + hi) / 2:.3f}%"})
		for d, (lo, hi) in enumerate(pairwise(day_edges), start=1)
		if hi > lo and (d - 1) % step == 0
	] + [html.I(className="deh-ev-tick", style={"left": f"{x:.3f}%"}) for x in day_edges[1:-1]]
	axis = [
		html.Span("Day", className="deh-ev-label deh-sub"),
		html.Div(
			[html.Div(className="deh-ev-band", style={"background": band(True)}), *ticks],
			className="deh-ev-axis",
		),
	]
	return place, day_phase, band, axis


def _events_card_children(recording: Recording) -> list:
	"""Every bout on one strip across the recording, then each event's bouts in full.

	Drawn from the config alone, as the habitat map is, so it renders before analysis. An
	event's colour is its place among the recording's events, the rule ``_event_spans``
	colours spans by, so a bar here matches its span on every plot.
	"""
	zone = recording.timeline.recording_timezone
	place, day_phase, band, axis = _strip_scale(recording.timeline)

	lanes, groups = [], []
	for index, event in enumerate(recording.events):
		color = qualitative.Pastel[index % len(qualitative.Pastel)]
		bars, rows = [], []
		for bout in sorted(event.bouts, key=lambda bout: bout.start):
			onset, offset = bout.start.astimezone(zone), bout.end.astimezone(zone)
			when = day_phase(onset)
			duration = _span(offset.astimezone(dt.UTC) - onset.astimezone(dt.UTC))
			where = ", ".join(map(_human, bout.position)) if bout.position else "Whole habitat"
			ends = (
				f"{offset:%H:%M}"
				if offset.date() == onset.date()
				else f"{offset.day} {offset:%b %H:%M}"
			)
			bars.append(
				html.Span(
					className="deh-ev-bar",
					style={
						**place(onset, offset),
						"background": color,
					},
					title=(
						f"{event.name} · {when}\n{onset.day} {onset:%b %H:%M} → {ends} "
						f"({duration})\n{where}"
					),
				)
			)
			rows.append(
				html.Tr(
					[
						html.Td(when),
						html.Td(
							f"{onset.day} {onset:%b %H:%M}",
							title=f"{onset:%a} {onset.day} {onset:%b %Y, %H:%M:%S} ({zone})",
						),
						html.Td(
							ends,
							title=f"{offset:%a} {offset.day} {offset:%b %Y, %H:%M:%S} ({zone})",
						),
						html.Td(duration, className="num"),
						html.Td(where),
					]
				)
			)

		lanes += [
			html.Span(
				[
					html.Span(className="deh-dot", style={"background": color}),
					html.Span(event.name, className="deh-ev-name"),
				],
				className="deh-ev-label",
				title=event.name,
			),
			html.Div(bars, className="deh-ev-track", style={"background": band(False)}),
		]
		groups.append(
			html.Tbody(
				[
					html.Tr(
						html.Td(
							html.Div(
								[
									html.Span(className="deh-dot", style={"background": color}),
									html.B(event.name),
									html.Span(event.description, className="deh-sub"),
								],
								className="deh-ev-label",
								role="button",
								tabIndex=0,
							),
							colSpan=5,
						),
						className="deh-ev-group",
					),
					*rows,
				]
			)
		)

	count, kinds = sum(len(event.bouts) for event in recording.events), len(recording.events)
	tally = f"{count} bout{'s' * (count != 1)} of {kinds} event{'s' * (kinds != 1)}"
	header = _card_head(
		"Events",
		[
			f"{tally}, as ",
			html.Code("config.json"),
			" declares them, over the light and dark phases they fell in.",
		],
		maximize=True,
	)
	strip = html.Div([*lanes, *axis], className="deh-ev-strip")
	table = html.Table(
		[
			html.Thead(
				html.Tr(
					[
						html.Th("Day · phase"),
						html.Th("Onset"),
						html.Th("Offset"),
						html.Th("Duration", className="num"),
						html.Th("Where"),
					]
				)
			),
			*groups,
		],
		className="deh-tbl mini deh-ev-table",
	)

	return [
		header,
		strip,
		html.Div(table, className="deh-cohort-table"),
		html.Footer(_reads(("events",)), className="deh-card-foot"),
	]


def _quality_missing_children(context: PlotContext, color_by: str) -> list:
	header = _card_head(
		"Position unknown",
		"Time each animal spent at a position antennas could not resolve, and how its "
		"passes were caught.",
		maximize=True,
	)
	needed = ("activity_df", "phase_durations", "recording_quality")
	missing = [table for table in needed if table not in context]
	if missing:
		return [header, _needs(*missing), html.Footer(_reads(needed), className="deh-card-foot")]

	result = services.missing_time(context)
	reads = services.reads_by_animal(context)
	subjects = (
		{row["animal_id"]: row["subject_name"] for row in context.animals.iter_rows(named=True)}
		if "animals" in context
		else {}
	)
	colors = resolve_colors(context, color_by).by_animal
	rows = [
		html.Tr(
			[
				html.Td(
					html.Span(className="deh-dot", style={"background": colors[row["animal_id"]]})
				),
				html.Td(row["animal_id"], className="deh-mono"),
				html.Td(subjects.get(row["animal_id"], "—")),
				html.Td(row["time_in_position_text"]),
				html.Td(f"{row['share']:.2f}%", className="num"),
				*(
					html.Td(f"{reads[row['animal_id']][kind]:,}", className="num")
					for kind in READ_KINDS
				),
			]
		)
		for row in result["rows"]
	]
	rows.append(
		html.Tr(
			[
				html.Td("Cohort mean · total", colSpan=4),
				html.Td(f"{result['mean_share']:.2f}%", className="num"),
				*(
					html.Td(f"{sum(animal[kind] for animal in reads.values()):,}", className="num")
					for kind in READ_KINDS
				),
			]
		)
	)
	table = html.Table(
		[
			html.Thead(
				html.Tr(
					[
						html.Th(""),
						html.Th("Tag"),
						html.Th("Subject"),
						html.Th("Time unknown"),
						html.Th("% of recording", className="num"),
						*(html.Th(kind.capitalize(), className="num") for kind in READ_KINDS),
					]
				)
			),
			html.Tbody(rows),
		],
		className="deh-tbl mini",
	)
	return [
		header,
		# Capped to the heatmap beside it, and scrolled past that, rather than letting a
		# large cohort set the height of the whole grid row.
		html.Div(table, className="deh-cohort-table", style={"maxHeight": f"{_MISSING_TABLE_H}px"}),
		html.Footer(_reads(needed), className="deh-card-foot"),
	]


def _habitat_card_children(context: PlotContext, height: int) -> list:
	assert context.recording is not None
	layout = context.recording.layout
	header = _card_head(
		"Habitat",
		[
			f"{len(layout.cages)} cages, {len(layout.tunnels)} tunnels and ",
			f"{len(topology.antennas(layout.antenna_combinations))} antennas, as ",
			html.Code("config.json"),
			" lays them out. Antennas are tinted by missed passes; click one for its reads.",
		],
		maximize=True,
	)

	# The map is drawn from the layout alone, so this card is the one that renders for a
	# recording whose pipeline has never run; the antennas just draw plain until there is a
	# quality table to band them by.
	antenna_stats = None
	reads = ("layout",)
	if "recording_quality" in context:
		antenna_stats = services.quality_summary(context)["antenna_stats"]
		reads = ("layout", "recording_quality")

	return [
		header,
		*components.habitat_map(
			layout,
			f"Habitat of {context.recording.name}",
			antenna_stats=antenna_stats,
			height=height,
		),
		html.Footer(_reads(reads), className="deh-card-foot"),
	]


def _card(cell: tuple, context: PlotContext, color_by: str, tab: str) -> html.Article:
	assert context.recording is not None
	name, span, height, *rest = cell
	rows = rest[0] if rest else 1
	match name:
		case "cohort":
			children = _cohort_card_children(context, color_by)
			return _card_frame(children, span, rows, card_id="cohort-card")
		case "habitat":
			return _card_frame(_habitat_card_children(context, height), span, rows)
		case "events":
			children = _events_card_children(context.recording)
			return _card_frame(children, span, rows, card_id="events-card")
		case "overview-summary":
			return _card_frame(_overview_summary_children(context), span, rows)
		case "quality-summary":
			return _card_frame(_quality_summary_children(context), span, rows)
		case "quality-missing":
			children = _quality_missing_children(context, color_by)
			return _card_frame(children, span, rows, card_id="quality-missing-card")
		case "sociability-heatmap":
			# One row of square pair matrices per FACETS_PER_LINE positions; either scope.
			facets = max(len(context.cages), len(context.tunnels))
			height *= math.ceil(facets / FACETS_PER_LINE)
	return _card_frame(_plot_card(name, context, height, tab), span, rows)


def _controls_bar(summary: dict, controls: dict, context: PlotContext) -> html.Div:
	bound = summary["days"] if controls["granularity"] == "day" else summary["phases"]
	attrs = available_attributes(context) if "animals" in context else ["animal_id"]
	color_by = controls["color_by"] if controls["color_by"] in attrs else "animal_id"
	cohort_children, cohort_pop = _cohort_widgets(context, color_by)
	one_phase = summary["phases"] == 1 or len(summary["onsets"]) == 1
	window_style = _HIDDEN if summary["phases"] == 1 else None
	animals_style = _HIDDEN if len(attrs) == 1 else None

	return html.Div(
		[
			html.Div(
				[
					html.Span("Window", className="deh-ctl-label"),
					dmc.SegmentedControl(
						id="rec-granularity",
						data=[
							{"value": "day", "label": "Days"},
							{"value": "phase_count", "label": "Phases"},
						],
						value=controls["granularity"],
						size="xs",
					),
				],
				className="deh-ctl",
				style=window_style,
			),
			html.Div(
				[
					html.Div(
						[
							html.Span(id="rec-window-label"),
							html.Span(id="rec-window-hint", className="deh-sub"),
						],
						className="deh-range-top",
					),
					dmc.RangeSlider(
						id="rec-window",
						min=1,
						max=bound,
						step=1,
						value=controls["window"],
						minRange=0,
						size="sm",
					),
				],
				className="deh-range-wrap",
				style=window_style,
			),
			html.Div(
				[
					html.Div(
						[
							html.Span(id="rec-hours-label"),
							html.Span(id="rec-hours-hint", className="deh-sub"),
						],
						className="deh-range-top",
					),
					html.Div(
						[
							html.Div(id="rec-hours-band", className="deh-band"),
							# Hours since the phase onset, as on the plots' hour axis, over
							# boundaries rather than bins so 0 → 12 reads as the first 12 h;
							# controls["hours"] stays in bins.
							dmc.RangeSlider(
								id="rec-hours",
								min=0,
								max=24,
								step=1,
								value=[controls["hours"][0], controls["hours"][1] + 1],
								marks=[{"value": h, "label": str(h)} for h in (0, 6, 12, 18, 24)],
								minRange=1,
								label=None,
								size="sm",
							),
						],
						className="deh-range",
					),
				],
				className="deh-range-wrap",
			),
			html.Div(
				[
					html.Span("Phases", className="deh-ctl-label"),
					dmc.ChipGroup(
						[
							dmc.Chip(
								[icon("sun", size=14), "Light"], value="light_phase", size="xs"
							),
							dmc.Chip(
								[icon("moon", size=14), "Dark"], value="dark_phase", size="xs"
							),
						],
						id="rec-phases",
						multiple=True,
						value=controls["phases"],
					),
				],
				className="deh-ctl",
				style=_HIDDEN if one_phase else None,
			),
			html.Div(
				[
					html.Span("Label", className="deh-ctl-label"),
					dmc.SegmentedControl(
						id="rec-label-by",
						data=[
							{"value": "animal_id", "label": "Tag"},
							{"value": "subject_name", "label": "Name"},
						],
						value=controls["label_by"],
						disabled="animals" not in context,
						size="xs",
					),
				],
				className="deh-ctl",
			),
			html.Div(
				[
					html.Span("Animals by", className="deh-ctl-label"),
					dmc.Select(
						id="rec-animals-by",
						data=[{"value": a, "label": _human(a)} for a in attrs],
						value=color_by,
						size="xs",
						w=150,
						allowDeselect=False,
					),
				],
				className="deh-ctl",
				style=animals_style,
			),
			dmc.Switch(
				id="rec-group-mean",
				label="Group mean",
				checked=controls.get("group_mean", False),
				disabled=color_by == "animal_id",
				size="xs",
				style=animals_style,
			),
			dmc.Switch(
				id="rec-events",
				label="Events",
				checked=False,
				size="xs",
				style=None if summary["events"] else _HIDDEN,
			),
			dmc.Popover(
				[
					dmc.PopoverTarget(
						html.Button(cohort_children, id="rec-cohort-btn", className="deh-btn sm")
					),
					dmc.PopoverDropdown(html.Div(cohort_pop, id="rec-cohort-pop")),
				],
				position="bottom-end",
				width=320,
				classNames={"dropdown": "deh-menu"},
			),
		],
		className="deh-controls",
	)


def _dashboard(
	pid: str,
	names: list[str],
	name: str,
	summary: dict,
	context: PlotContext,
	controls: dict,
	tab: str,
) -> list:
	header = html.Div(
		[
			html.Div(
				[
					html.Button(
						icon("chevron-left", size=16),
						id="rec-prev",
						className="deh-icon-btn",
						title="Previous recording",
						disabled=len(names) < 2,
					),
					dmc.Select(
						id="rec-select",
						data=[{"value": n, "label": n} for n in names],
						value=name,
						size="sm",
						w=270,
						allowDeselect=False,
						searchable=True,
						classNames={"input": "deh-mono"},
					),
					html.Button(
						icon("chevron-right", size=16),
						id="rec-next",
						className="deh-icon-btn",
						title="Next recording",
						disabled=len(names) < 2,
					),
				],
				className="deh-rec-pick",
			),
			# Before the facts: a float only rises to the line it comes after.
			components.download_menu(
				pid,
				name,
				html.Button([icon("download", size=16), "Download"], className="deh-btn sm"),
				[table for table in DataFrameRegistry.list_available() if table in context],
			),
			html.Div(_meta_strip(summary), className="deh-rec-meta"),
		],
		className="deh-rec-head",
	)
	tabs = dmc.Tabs(
		[
			dmc.TabsList(
				[dmc.TabsTab(label, value=section_id) for section_id, label, _ in _SECTIONS]
			),
			*(
				dmc.TabsPanel(
					html.Div(
						[
							_card(cell, context, controls["color_by"], section_id)
							for cell in cells
							if cell[0] != "events" or summary["events"]
						],
						className="deh-cards",
					),
					value=section_id,
				)
				for section_id, _, cells in _SECTIONS
			),
		],
		id="rec-tabs",
		value=tab,
		# Dash 4 rebuilds a remounted panel from the layout its parent first rendered,
		# which drops every figure a callback has filled in since.
		keepMounted=True,
		className="deh-tabs-gap",
	)
	return [header, _controls_bar(summary, controls, context), tabs]


@callback(
	Output("rec-body", "children"),
	Output("rec-context", "data"),
	Output("rec-controls", "data", allow_duplicate=True),
	Input("url", "pathname"),
	Input("url", "search"),
	State("project-paths", "data"),
	State("rec-context", "data"),
	prevent_initial_call="initial_duplicate",
)
def _resolve(pathname, search, paths, current):
	if pathname != PATH:
		raise PreventUpdate

	params = parse_qs((search or "").lstrip("?"))
	pid = (params.get("project") or [None])[0]
	requested = (params.get("recording") or [None])[0]
	tab = (params.get("tab") or [_SECTIONS[0][0]])[0]
	tab = tab if tab in _TAB_IDS else _SECTIONS[0][0]

	if not pid:
		return (
			components.placeholder("Open a recording from a project to see its dashboard here."),
			None,
			no_update,
		)

	location = services.resolve_project(paths, pid)
	if location is None:
		message = "This project is not in your browser's saved list. Open it from Projects."
		return components.placeholder(message), None, no_update

	try:
		project = services.load_project(location)
	except Exception as exc:
		return components.placeholder(f"{type(exc).__name__}: {exc}"), None, no_update

	names = [recording.name for recording in project.recordings]
	if not names:
		return (
			components.placeholder(f"{project.project_name} has no recordings yet."),
			None,
			no_update,
		)

	name = requested if requested in names else names[0]
	# A string: nanosecond mtimes lose precision as JSON numbers in the browser.
	stamp = str(services.results_stamp(location, name))

	identity = {"location": location, "recording": name, "stamp": stamp}
	if current and all(current.get(key) == value for key, value in identity.items()):
		raise PreventUpdate

	try:
		summary = services.recording_summary(location, name)
		context = services.plot_context(location, name)
	except Exception as exc:
		return components.placeholder(f"{type(exc).__name__}: {exc}"), None, no_update

	controls = {
		"window": [1, summary["days"]],
		"granularity": "day",
		"hours": [0, 23],
		"phases": list(PHASES),
		"color_by": "animal_id",
		"label_by": "animal_id",
		"group_mean": False,
	}
	body = _dashboard(pid, names, name, summary, context, controls, tab)
	# days/phases ride along so the window slider can re-axis itself clientside.
	context_data = {
		**identity,
		"days": summary["days"],
		"phases": summary["phases"],
		# The hours band and its hover text are painted clientside.
		"onsets": summary["onsets"],
		"start_from": summary["start_from"],
	}
	return body, context_data, controls


def _merge_search(search: str, **updates: str) -> str:
	params = {key: values[0] for key, values in parse_qs((search or "").lstrip("?")).items()}
	params.update(updates)
	return "?" + urlencode(params, quote_via=quote)


@callback(
	Output("url", "search", allow_duplicate=True),
	# dcc.Location pushes a changed href ahead of a changed search, and its own href still holds
	# the URL it last saw, which deh.switchTab's replaceState has since moved past: sent alone,
	# the search would lose to that stale href.
	Output("url", "href", allow_duplicate=True),
	Input("rec-select", "value"),
	Input("rec-prev", "n_clicks"),
	Input("rec-next", "n_clicks"),
	State("rec-context", "data"),
	State("url", "search"),
	# url.search lags on the tab (see deh.switchTab), so the showing one comes from here.
	State("rec-tab", "data"),
	prevent_initial_call=True,
)
def _switch_recording(selected, _prev, _next, context_data, search, tab):
	# All three inputs are rebuilt into rec-body on every recording switch, so they fire this
	# once with no click and no trigger behind them (see _notes_modal). Unguarded, that lands
	# in the step branch as step -1 and walks to the previous recording on its own, which
	# rebuilds the body and fires it again.
	if not context_data or not ctx.triggered[0]["value"]:
		raise PreventUpdate

	location = context_data["location"]
	names = [recording.name for recording in services.load_project(location).recordings]
	current = context_data["recording"]

	if ctx.triggered_id == "rec-select":
		name = selected
	else:
		step = 1 if ctx.triggered_id == "rec-next" else -1
		name = names[(names.index(current) + step) % len(names)]

	if name == current:
		raise PreventUpdate

	search = _merge_search(search, recording=name, **({"tab": tab} if tab else {}))
	return search, PATH + search


clientside_callback(
	ClientsideFunction("deh", "switchTab"),
	Output("rec-tab", "data"),
	Input("rec-tabs", "value"),
	prevent_initial_call=True,
)


clientside_callback(
	ClientsideFunction("deh", "paintHabitat"),
	Input("rec-context", "data"),
)


clientside_callback(
	ClientsideFunction("deh", "tabJump"),
	Input("rec-habitat-jump", "n_clicks"),
	Input("rec-quality-jump", "n_clicks"),
	Input("rec-mice-jump", "n_clicks"),
	Input("rec-events-jump", "n_clicks"),
	prevent_initial_call=True,
)


clientside_callback(
	ClientsideFunction("deh", "windowControl"),
	Output("rec-window", "min"),
	Output("rec-window", "max"),
	Output("rec-window", "marks"),
	Output("rec-window", "value"),
	Output("rec-controls", "data", allow_duplicate=True),
	Input("rec-granularity", "value"),
	Input("rec-window", "value"),
	State("rec-context", "data"),
	State("rec-controls", "data"),
	# Still runs whenever rec-body mounts, as rec-controls sits outside it: Dash fires a
	# callback whose Inputs arrive in a new chunk if any Output lies outside that chunk.
	prevent_initial_call=True,
)


clientside_callback(
	ClientsideFunction("deh", "filterControls"),
	Output("rec-controls", "data", allow_duplicate=True),
	Output("rec-group-mean", "disabled"),
	Output("rec-hours", "value"),
	Output("rec-phases", "value"),
	Input("rec-hours", "value"),
	Input("rec-phases", "value"),
	Input("rec-animals-by", "value"),
	Input("rec-label-by", "value"),
	Input("rec-group-mean", "checked"),
	State("rec-controls", "data"),
	State("rec-context", "data"),
	# Runs whenever rec-body mounts too, like windowControl above.
	prevent_initial_call=True,
)


clientside_callback(
	ClientsideFunction("deh", "toggleEvents"),
	Input("rec-events", "checked"),
	State({"type": "plot", "plot": ALL}, "figure"),
	State({"type": "plot", "plot": ALL}, "id"),
	prevent_initial_call=True,
)


clientside_callback(
	ClientsideFunction("deh", "colorBy"),
	Output("rec-color-by", "data"),
	Input("rec-controls", "data"),
	State("rec-color-by", "data"),
	prevent_initial_call=True,
)


@callback(
	Output("rec-cohort-btn", "children"),
	Output("rec-cohort-pop", "children"),
	Output("cohort-card", "children"),
	Output("quality-missing-card", "children"),
	Input("rec-color-by", "data"),
	State("rec-context", "data"),
	prevent_initial_call=True,
)
def _update_cohort(color_by, context_data):
	if not context_data:
		raise PreventUpdate
	context = services.plot_context(context_data["location"], context_data["recording"])
	return (
		*_cohort_widgets(context, color_by),
		_cohort_card_children(context, color_by),
		_quality_missing_children(context, color_by),
	)


def _cohort_card_children(context: PlotContext, color_by: str) -> list:
	header = _card_head(
		"Cohort",
		[
			f"{len(context.animal_ids)} animals. Colour follows ",
			html.B(_human(color_by)),
			" on every plot here.",
		],
		maximize=True,
	)
	if "animals" not in context:
		return [header, _needs("animals")]

	mapping, animals = resolve_colors(context, color_by), context.animals.sort("animal_id")
	table = html.Table(
		[
			html.Thead(
				html.Tr(
					[
						html.Th(""),
						html.Th("Tag"),
						html.Th("Subject"),
						html.Th("Sex"),
						html.Th("Genotype"),
						html.Th("Mouse line"),
						html.Th("Genetic background"),
						html.Th("Treatment"),
						html.Th("Age", className="num"),
					]
				)
			),
			html.Tbody(
				[
					html.Tr(
						[
							html.Td(
								html.Span(
									className="deh-dot",
									style={"background": mapping.by_animal[row["animal_id"]]},
								)
							),
							html.Td(row["animal_id"], className="deh-mono"),
							html.Td(row["subject_name"]),
							html.Td(row["sex"]),
							html.Td(row["genotype"]),
							html.Td(row["mouse_line"]),
							html.Td(row["genetic_background"]),
							html.Td(row["treatment"]),
							html.Td(f"{row['age'].days} d", className="num"),
						],
						id={"type": "animal-row", "scope": "table", "tag": row["animal_id"]},
						n_clicks=0,
						style={"cursor": "pointer"},
						title="Edit notes",
					)
					for row in animals.iter_rows(named=True)
				]
			),
		],
		className="deh-tbl mini",
	)
	return [
		header,
		html.Div(table, className="deh-cohort-table"),
		html.Footer(_reads(("animals",)), className="deh-card-foot"),
	]


clientside_callback(
	ClientsideFunction("deh", "plotRequest"),
	Input("rec-context", "data"),
	Input("rec-controls", "data"),
	Input("rec-tab", "data"),
	Input("plot-theme", "data"),
	Input({"type": "card-opt", "plot": ALL, "option": ALL}, "value"),
	Input("rec-hover", "data"),
	State({"type": "plot-req", "plot": ALL}, "data"),
	State({"type": "badge", "plot": ALL, "control": ALL}, "hidden"),
)


@callback(
	Output({"type": "plot", "plot": MATCH}, "figure"),
	Input({"type": "plot-req", "plot": MATCH}, "data"),
	State("rec-events", "checked"),
	# The request mounts as {tab} alone; deh.plotRequest fills it once the tab shows.
	prevent_initial_call=True,
)
def _update_plot(request, events_on):
	if not request or "context" not in request:
		raise PreventUpdate

	return _build_plot(ctx.outputs_list["id"]["plot"], request, events_on)


clientside_callback(
	ClientsideFunction("deh", "timelineZoom"),
	Output("rec-timeline-zoom", "data"),
	Input({"type": "plot", "plot": "recording-timeline"}, "relayoutData"),
	State("rec-timeline-zoom", "data"),
	prevent_initial_call=True,
)


@callback(
	Output({"type": "plot", "plot": "recording-timeline"}, "figure", allow_duplicate=True),
	Input("rec-timeline-zoom", "data"),
	State({"type": "plot-req", "plot": "recording-timeline"}, "data"),
	State("rec-events", "checked"),
	prevent_initial_call=True,
)
def _zoom_timeline(x_range, request, events_on):
	if not request or "context" not in request:
		raise PreventUpdate

	return _build_plot("recording-timeline", request, events_on, x_range=x_range)


def _build_plot(name: str, request: dict, events_on: bool, **extra) -> go.Figure:
	context_data, controls, theme = request["context"], request["controls"], request["theme"]
	context = services.plot_context(context_data["location"], context_data["recording"])
	spec = PlotRegistry.spec(name)
	if any(table not in context for table in spec.requires):
		raise PreventUpdate

	# A cleared NumberInput sends "", which falls back to the builder's default.
	values = {key: value for key, value in request["opts"].items() if value != ""}
	# deh.plotRequest sends only the controls this plot takes.
	values.update(
		(option, controls[key]) for option, key in _GLOBAL_OPTIONS.items() if key in controls
	)
	# None for the untouched full day, matching every builder's own default: some prepare.*
	# steps aggregate away the hour column before applying this filter and only skip that
	# filter when it is exactly None (e.g. prep_polar).
	if values.get("hours_range") == [0, 23]:
		values["hours_range"] = None

	figure = PlotRegistry.build(name, context, **values, **extra)
	# The card header already carries the title; the figure's own would double it up.
	# Plotly still reserves top margin for it, so that's trimmed back too - which is also
	# where the phase band sits, so it keeps a little more room than the title needed.
	# Rebuilt for the same request - a zoomed timeline - the figure keeps the viewer's
	# zoom and hidden legend entries; a new request starts afresh.
	figure.update_layout(
		title=None, margin={"t": 30}, uirevision=hash(json.dumps(request, sort_keys=True))
	)
	# Not a bare template=: a shape carries a literal colour, so the phase band has to be
	# repainted for the theme rather than inheriting it.
	plot_theme.apply(figure, theme or "light")
	if not events_on:
		# A newly mounted or rebuilt card starts respecting the toggle without being an
		# Input of its own; deh.toggleEvents keeps an already-loaded figure in sync.
		figure.update_shapes(visible=False, selector={"name": "event-span"})
		figure.update_shapes(visible=False, selector={"name": "event-label"})
	return figure


# Clientside: every figure on the tab is already in the browser, so picking one is not
# worth uploading them all to the server and sending one back.
clientside_callback(
	ClientsideFunction("deh", "fullscreen"),
	Output("fullscreen-modal", "opened"),
	Output("fullscreen-modal", "title"),
	Output("fullscreen-plot", "figure"),
	Output("fullscreen-plot", "style"),
	Output("fullscreen-frame", "style"),
	Input({"type": "card-fullscreen", "plot": ALL}, "n_clicks"),
	State({"type": "plot", "plot": ALL}, "figure"),
	State({"type": "plot", "plot": ALL}, "id"),
	State({"type": "card-title", "plot": ALL}, "children"),
	State({"type": "card-title", "plot": ALL}, "id"),
	prevent_initial_call=True,
)


# Clientside for the same reason as the fullscreen callback above. The export-source store
# still feeds the server preview, so one figure goes up - not all of them.
clientside_callback(
	ClientsideFunction("deh", "openExport"),
	Output("export-dialog", "opened", allow_duplicate=True),
	Output("export-source", "data", allow_duplicate=True),
	Output("export-filename", "value", allow_duplicate=True),
	Input({"type": "card-export", "plot": ALL}, "n_clicks"),
	State({"type": "plot", "plot": ALL}, "figure"),
	State({"type": "plot", "plot": ALL}, "id"),
	State({"type": "card-title", "plot": ALL}, "children"),
	State({"type": "card-title", "plot": ALL}, "id"),
	State("rec-context", "data"),
	prevent_initial_call=True,
)


# Format only rewrites layout the browser already holds, so none of it reaches the server.
clientside_callback(
	ClientsideFunction("deh", "openFormat"),
	Output("rec-format-modal", "opened"),
	Output("rec-format-modal", "title"),
	Output("rec-format-target", "data"),
	Output({"type": "rec-fmt", "key": ALL}, "value"),
	Output({"type": "rec-fmt", "key": ALL}, "placeholder"),
	Output({"type": "rec-fmt", "key": ALL}, "disabled"),
	Output({"type": "rec-fmt", "key": "palette"}, "data"),
	Input({"type": "card-format", "plot": ALL}, "n_clicks"),
	State({"type": "plot", "plot": ALL}, "figure"),
	State({"type": "plot", "plot": ALL}, "id"),
	State({"type": "card-title", "plot": ALL}, "children"),
	State({"type": "card-title", "plot": ALL}, "id"),
	State("rec-format", "data"),
	State("rec-colors", "data"),
	prevent_initial_call=True,
)


clientside_callback(
	ClientsideFunction("deh", "editFormat"),
	Output("rec-format", "data"),
	Input({"type": "rec-fmt", "key": ALL}, "value"),
	Input("rec-fmt-reset", "n_clicks"),
	State("rec-format-target", "data"),
	State("rec-format", "data"),
	State({"type": "plot", "plot": ALL}, "figure"),
	State({"type": "plot", "plot": ALL}, "id"),
	State("rec-colors", "data"),
	prevent_initial_call=True,
)


clientside_callback(
	ClientsideFunction("deh", "applyFormat"),
	Input("rec-format", "data"),
	Input({"type": "plot", "plot": ALL}, "figure"),
	State({"type": "plot", "plot": ALL}, "id"),
	State("rec-colors", "data"),
	prevent_initial_call=True,
)


@callback(
	Output("notes-modal", "opened"),
	Output("notes-modal", "title"),
	Output("notes-textarea", "value"),
	Output("notes-target", "data"),
	Input("rec-notes-btn", "n_clicks"),
	Input({"type": "animal-row", "scope": ALL, "tag": ALL}, "n_clicks"),
	Input("notes-cancel", "n_clicks"),
	Input("notes-save", "n_clicks"),
	State("notes-textarea", "value"),
	State("rec-context", "data"),
	State("notes-target", "data"),
	prevent_initial_call=True,
)
def _notes_modal(_open, _rows, _cancel, _save, text, context_data, target):
	# rec-notes-btn and the animal rows are rebuilt into rec-body on every recording
	# switch, and notes-cancel / notes-save live in the page's static layout, which
	# mounts fresh whenever this page is routed to - all fire this callback once with
	# no real click behind them.
	if not ctx.triggered[0]["value"]:
		raise PreventUpdate
	trigger = ctx.triggered_id
	if trigger == "notes-cancel":
		return False, no_update, no_update, no_update

	if not context_data:
		raise PreventUpdate
	location, name = context_data["location"], context_data["recording"]

	if trigger == "notes-save":
		if not target:
			raise PreventUpdate
		services.update_notes(location, name, (text or "").strip(), target.get("tag"))
		notify("good", "Notes saved")
		return False, no_update, no_update, no_update

	# Read fresh from disk rather than context.animals, the analysed parquet, which
	# would go stale the moment notes are saved without a re-analysis.
	recording = services.load_project(location)[name]
	if trigger == "rec-notes-btn":
		return True, "Notes", recording.notes, {"tag": None}

	tag = trigger["tag"]
	animal = next(a for a in recording.cohort.animals if a.tag == tag)
	return True, f"{animal.subject_name} · {tag}", animal.notes, {"tag": tag}
