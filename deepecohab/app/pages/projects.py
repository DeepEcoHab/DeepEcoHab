import base64
import io
import json
import shutil
import tempfile
import time
from pathlib import Path
from urllib.parse import quote

import dash
import dash_mantine_components as dmc
import polars as pl
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
	set_props,
)
from dash.exceptions import PreventUpdate
from pydantic import ValidationError

from deepecohab import AnalysisParams, Project, Recording
from deepecohab.app import components, services
from deepecohab.app.components import icon, notify

dash.register_page(__name__, path="/", name="Projects", order=0, icon="folders")

_DEFAULTS = AnalysisParams()
_PROJECTS_HOME = Path.home() / "Documents" / "deepecohab" / "projects"


def _field(title: str, name: str, control, help_text: str) -> html.Div:
	return html.Div(
		[html.Span([title, " ", html.Code(name)]), control, html.P(help_text)],
		className="deh-field",
	)


def _seconds(control_id: str, value: float) -> html.Span:
	return html.Span(
		[
			dmc.NumberInput(
				id=control_id,
				value=value,
				min=0,
				step=0.5,
				allowNegative=False,
				w=96,
				classNames={"input": "deh-input"},
			),
			html.Span("seconds", className="deh-sub"),
		],
		className="deh-unit-input",
	)


layout = html.Div(
	[
		dcc.Store(id="expanded", data=[]),
		dcc.Store(id="selection", data=[]),
		dcc.Store(id="analysis-params"),
		dcc.Store(
			id="params-defaults",
			data={
				"minimum_time": _DEFAULTS.minimum_time,
				"minimum_time_alone": _DEFAULTS.minimum_time_alone,
				"chasing_time_window": list(_DEFAULTS.chasing_time_window),
			},
		),
		dcc.Store(id="run-progress"),
		dcc.Store(id="run-active", data=False),
		dcc.Store(id="run-failed", data={}),
		dcc.Store(id="data-changed"),
		dcc.Store(id="upload-target"),
		dcc.Store(id="replace-pending"),
		dcc.Store(id="replace-queue"),
		dcc.Store(id="remove-target"),
		# Real clicks on the table's row buttons, via deh.clickEvent: the table is rebuilt on
		# every search or data change, which would otherwise fire their callbacks each time.
		dcc.Store(id="remove-project-event"),
		dcc.Store(id="generate-table-event"),
		dcc.Store(id="add-recordings-event"),
		dcc.Store(id="remove-recording-event"),
		dcc.Store(id="remove-recordings-event"),
		dcc.Store(id="reinstate-recordings-event"),
		html.Div(
			[
				html.Div([html.H2("Projects"), html.P(id="project-count")], className="deh-titles"),
				dmc.TextInput(
					id="project-search",
					placeholder="Filter projects and recordings",
					leftSection=icon("search", size=16),
					debounce=150,
					className="deh-search",
					classNames={"input": "deh-input"},
					**{"aria-label": "Filter projects and recordings"},
				),
				html.Button(
					[icon("plus", size=16), "Add project"],
					id="add-open",
					className="deh-btn",
				),
				html.Button(
					[icon("folder-plus", size=16), "New project"],
					id="create-open",
					className="deh-btn deh-btn-primary",
				),
			],
			className="deh-screen-head",
		),
		html.Div(id="project-table", className="deh-table-wrap"),
		html.Div(
			[
				html.Div(id="actionbar-message", className="deh-ab-msg"),
				html.Div(html.I(id="actionbar-meter"), className="deh-meter deh-when-running"),
				html.Button(
					[icon("format", size=16), "Parameters"],
					id="params-open",
					className="deh-btn deh-when-idle",
				),
				dmc.Switch(
					id="overwrite",
					checked=False,
					label="Overwrite existing tables",
					size="sm",
					className="deh-switch deh-when-idle",
				),
				dmc.Switch(
					id="use-prev-ranking",
					checked=True,
					label="Use stored previous rankings",
					size="sm",
					className="deh-switch deh-when-idle deh-when-overwrite",
				),
				html.Span(className="deh-grow"),
				html.Button(
					"Clear", id="selection-clear", className="deh-btn deh-btn-ghost deh-when-idle"
				),
				html.Button(
					[icon("player-play", size=16), "Run analysis"],
					id="run-start",
					className="deh-btn deh-btn-primary deh-when-idle",
				),
				html.Button("Cancel", id="run-cancel", className="deh-btn deh-when-running"),
			],
			id="actionbar",
			className="deh-actionbar",
			hidden=True,
		),
		dmc.Drawer(
			id="params-drawer",
			title="Analysis parameters",
			position="right",
			size="380px",
			classNames=components.DIALOG_CLASSES,
			children=[
				html.Div(
					[
						_field(
							"Minimum meeting",
							"minimum_time",
							_seconds("params-minimum-time", _DEFAULTS.minimum_time),
							"Minimum continuous co-presence for a meeting to count. "
							"0 keeps every meeting.",
						),
						_field(
							"Minimum time alone",
							"minimum_time_alone",
							_seconds("params-minimum-alone", _DEFAULTS.minimum_time_alone),
							"Minimum solitary span. Discards the brief gaps left by animals "
							"arriving moments apart.",
						),
						_field(
							"Chasing window",
							"chasing_time_window",
							dmc.RangeSlider(
								id="params-chasing",
								value=list(_DEFAULTS.chasing_time_window),
								min=0,
								max=5,
								step=0.1,
								minRange=0.1,
								precision=1,
								marks=[
									{"value": second, "label": f"{second} s"} for second in range(6)
								],
								mb="lg",
							),
							"Shortest and longest chasing event, in seconds.",
						),
						_field(
							"Previous ranking",
							Recording.PREV_RANKING,
							html.Div(
								[
									dcc.Upload(
										id="params-ranking-upload",
										multiple=True,
										accept=".parquet,.csv",
										className="deh-drop",
										className_active="deh-drop is-over",
										children=[
											icon("upload", size=20),
											html.B("Drop ranking tables, or click to pick them"),
											html.Span(
												[
													"An earlier ",
													html.Code("ranking.parquet"),
													" renamed ",
													html.Code("<recording>.parquet"),
													(
														" is stored with that recording, as each "
														"animal's last rating."
													),
												]
											),
										],
									),
									html.Div(
										[
											html.Span(
												id="params-ranking-status", className="deh-sub"
											),
											html.Button(
												icon("x", size=13),
												id="params-ranking-clear",
												className="deh-icon-btn sm",
												title="Remove the stored rankings of the selection",
											),
										],
										className="deh-unit-input",
									),
								]
							),
							"Starts a recording's dominance ranking from an earlier recording of "
							"the same animals, instead of from scratch, whenever it is built.",
						),
					],
					className="deh-dialog-body",
				),
				html.Div(
					[
						html.Button(
							"Reset to defaults",
							id="params-reset",
							className="deh-btn deh-btn-ghost",
						),
						html.Button(
							"Apply", id="params-apply", className="deh-btn deh-btn-primary"
						),
					],
					className="deh-dialog-foot",
				),
			],
		),
		dmc.Modal(
			id="add-modal",
			title="Add project",
			size=520,
			classNames=components.DIALOG_CLASSES,
			children=[
				html.Div(
					dmc.TextInput(
						id="add-path",
						label="Project folder",
						description=[
							"The folder that holds ",
							html.Code("project.json"),
							". The path is remembered in this browser only; files are never moved.",
						],
						placeholder=r"C:\Users\you\Documents\deepecohab\projects\my_project",
						autoComplete="off",
						inputWrapperOrder=["label", "input", "description", "error"],
						className="deh-field",
						classNames={"input": "deh-input deh-mono"},
					),
					className="deh-dialog-body",
				),
				html.Div(
					[
						html.Button("Cancel", id="add-cancel", className="deh-btn deh-btn-ghost"),
						html.Button(
							"Add project", id="add-submit", className="deh-btn deh-btn-primary"
						),
					],
					className="deh-dialog-foot",
				),
			],
		),
		dmc.Modal(
			id="create-modal",
			title="New project",
			size=520,
			classNames=components.DIALOG_CLASSES,
			children=[
				html.Div(
					[
						dmc.TextInput(
							id="create-name",
							label="Project name",
							placeholder="my_project",
							className="deh-field",
							classNames={"input": "deh-input"},
						),
						dmc.TextInput(
							id="create-experimenter",
							label="Experimenter",
							placeholder="Who runs the experiment",
							className="deh-field",
							classNames={"input": "deh-input"},
						),
						dmc.TextInput(
							id="create-path",
							label="Folder",
							description="The project gets its own folder, named after it, in here. "
							"Leave empty to use the folder shown.",
							placeholder=str(_PROJECTS_HOME),
							autoComplete="off",
							inputWrapperOrder=["label", "input", "description", "error"],
							className="deh-field",
							classNames={"input": "deh-input deh-mono"},
						),
						dmc.Textarea(
							id="create-description",
							label="Description",
							placeholder="Optional: cohorts, treatment, what the project is for",
							autosize=True,
							minRows=2,
							className="deh-field",
						),
					],
					className="deh-dialog-body",
				),
				html.Div(
					[
						html.Button(
							"Cancel", id="create-cancel", className="deh-btn deh-btn-ghost"
						),
						html.Button(
							"Create project",
							id="create-submit",
							className="deh-btn deh-btn-primary",
						),
					],
					className="deh-dialog-foot",
				),
			],
		),
		dmc.Modal(
			id="upload-modal",
			title="Add recordings",
			size=560,
			classNames=components.DIALOG_CLASSES,
			children=[
				html.Div(
					[
						dcc.Upload(
							id="upload-files",
							multiple=True,
							accept=".json,.parquet",
							className="deh-drop",
							className_active="deh-drop is-over",
							children=[
								icon("upload", size=22),
								html.B("Drop recordings here, or click to pick them"),
								html.Span(
									[
										"One ",
										html.Code("<name>.config.json"),
										" beside its ",
										html.Code("<name>.data.parquet"),
										" and ",
										html.Code("<name>.diagnostic.json"),
										(
											". Files are copied into the project under the name "
											"the config carries."
										),
									]
								),
							],
						),
						html.Div(id="upload-report"),
					],
					className="deh-dialog-body",
				),
				html.Div(
					html.Button("Close", id="upload-close", className="deh-btn"),
					className="deh-dialog-foot",
				),
			],
		),
		dmc.Modal(
			id="replace-modal",
			title="Replace recording?",
			size=520,
			closeOnClickOutside=False,
			closeOnEscape=False,
			withCloseButton=False,
			classNames=components.DIALOG_CLASSES,
			children=[
				html.Div(
					[
						html.P(id="replace-text"),
						dmc.Checkbox(id="replace-all", size="xs", radius="xs"),
					],
					className="deh-dialog-body",
				),
				html.Div(
					[
						html.Button("Skip", id="replace-skip", className="deh-btn deh-btn-ghost"),
						html.Button(
							"Replace", id="replace-confirm", className="deh-btn deh-btn-danger"
						),
					],
					className="deh-dialog-foot",
				),
			],
		),
		dmc.Modal(
			id="remove-modal",
			title="Remove recording",
			size=520,
			classNames=components.DIALOG_CLASSES,
			children=[
				html.Div(
					html.P(
						"Delisting takes the recording out of the project and leaves every file "
						"where it is, so Reinstate brings it back. Deleting also removes "
						"the recording's folder — its config, raw registrations and results "
						"— for good."
					),
					className="deh-dialog-body",
				),
				html.Div(
					[
						html.Button(
							"Cancel", id="remove-cancel", className="deh-btn deh-btn-ghost"
						),
						html.Button("Delist, keep files", id="remove-delist", className="deh-btn"),
						html.Button(
							[icon("trash", size=16), "Delete files"],
							id="remove-delete",
							className="deh-btn deh-btn-danger",
						),
					],
					className="deh-dialog-foot",
				),
			],
		),
	]
)


def _visible_recordings(project: dict, query: str) -> list[dict]:
	"""A project's recordings under the search; all of them when the project name matches."""
	if not query or query in project["name"].lower():
		return project["recordings"]
	return [recording for recording in project["recordings"] if query in recording["name"].lower()]


def _badge(kind: str, icon_name: str, *content, spin: bool = False, title: str = "") -> html.Span:
	return html.Span(
		[icon(icon_name, size=14, class_name="deh-spin" if spin else ""), *content],
		className=f"deh-badge deh-badge-{kind}",
		title=title,
	)


def _table_cell(project: dict) -> html.Span | html.Div:
	rows = project["table_rows"]
	if rows is None:
		return _badge("neutral", "circle-dashed", "Not generated")
	if not project["table_stale"]:
		return _badge("neutral", "database", f"{rows:,} rows")
	return html.Div(
		[
			_badge(
				"warn",
				"alert-triangle",
				f"{rows:,} rows · regenerate",
				title="Recordings or results changed since the project table was built",
			),
			html.Button(
				icon("refresh", size=15),
				id={"type": "generate-table", "place": "cell", "index": project["location"]},
				className="deh-icon-btn sm",
				title="Regenerate the project table",
			),
		],
		className="deh-table-cell",
	)


def _running_badge(running: list) -> html.Span:
	steps, total, building = running
	building = html.Span(building or "done", className="deh-mono")
	return _badge("run", "loader-2", f"{steps}/{total} · ", building, spin=True)


def _status(recording: dict, running: list | None, failure: str | None) -> html.Span:
	done, total = recording["done"], recording["total"]
	if running is not None:
		return _running_badge(running)
	if failure is not None:
		return _badge("bad", "circle-x", "Failed", title=failure)
	if done == total:
		return _badge("good", "circle-check", "Analysed", html.Small(f"{done}/{total}"))
	if done == 0:
		return _badge("neutral", "circle-dashed", "Not analysed", html.Small(f"0/{total}"))
	return _badge("warn", "alert-triangle", "Partial", html.Small(f"{done}/{total}"))


def _project_menu(project: dict) -> dmc.Menu:
	location, loadable = project["location"], project["error"] is None
	return dmc.Menu(
		[
			dmc.MenuTarget(
				html.Button(
					icon("dots-vertical"),
					className="deh-icon-btn",
					title=f"Actions for {project['name']}",
				)
			),
			dmc.MenuDropdown(
				[
					dmc.MenuLabel(project["name"]),
					dmc.MenuItem(
						"Open in plot builder",
						leftSection=icon("drag-drop", size=16),
						href=dash.get_relative_path(f"/builder?project={project['id']}"),
						disabled=not loadable,
					),
					dmc.MenuItem(
						"Add recordings…",
						leftSection=icon("upload", size=16),
						id={"type": "add-recordings", "place": "menu", "index": location},
						disabled=not loadable,
					),
					dmc.MenuItem(
						"Generate project table",
						leftSection=icon("database", size=16),
						id={"type": "generate-table", "place": "menu", "index": location},
						disabled=not loadable,
					),
					*(
						[
							dmc.MenuDivider(),
							dmc.MenuLabel("Download"),
							*(
								dmc.MenuItem(
									label,
									leftSection=icon(icon_name, size=16),
									href=href,
									target=components.DOWNLOAD_FRAME,  # ty: ignore[invalid-argument-type]
								)
								for icon_name, label, href in _project_download_items(project["id"])
							),
						]
						if loadable
						else []
					),
					dmc.MenuItem(
						"Copy path",
						leftSection=icon("copy", size=16),
						id={"type": "copy-path", "place": "menu", "index": location},
					),
					dmc.MenuDivider(),
					dmc.MenuItem(
						"Remove from list",
						leftSection=icon("trash", size=16),
						id={"type": "remove-project", "index": location},
						className="deh-danger",
					),
				]
			),
		],
		position="bottom-end",
		classNames={"dropdown": "deh-menu"},
	)


def _reinstate_menu(project: dict) -> dmc.Popover:
	location, delisted = project["location"], project["delisted"]
	return dmc.Popover(
		[
			dmc.PopoverTarget(
				html.Button(
					[icon("arrow-back-up", size=16), f"Reinstate ({len(delisted)})"],
					className="deh-btn deh-btn-ghost sm",
					title="Bring back delisted recordings, with their results",
					disabled=project["error"] is not None,
				)
			),
			dmc.PopoverDropdown(
				dmc.Stack(
					[
						dcc.Store(id={"type": "reinstate-names", "index": location}, data=delisted),
						dmc.Checkbox(
							id={"type": "reinstate-all", "index": location},
							label="Select all",
							checked=False,
							size="xs",
						),
						dmc.CheckboxGroup(
							id={"type": "reinstate-pick", "index": location},
							value=[],
							children=dmc.Stack(
								[
									dmc.Checkbox(
										value=name,
										label=html.Span(name, className="deh-mono"),
										size="xs",
									)
									for name in delisted
								],
								gap=6,
							),
						),
						html.Button(
							"Reinstate",
							id={"type": "reinstate-recordings", "index": location},
							className="deh-btn deh-btn-primary sm",
						),
					],
					gap=10,
				)
			),
		],
		position="bottom-start",
		classNames={"dropdown": "deh-menu"},
	)


def _project_download_items(pid: str) -> list[tuple[str, str, str]]:
	base = f"/download/project/{pid}"
	return [
		("table", "Project table · parquet", f"{base}/table"),
		("file-type-csv", "Project table · CSV", f"{base}/table?format=csv"),
		("file-zip", "Configs and results · zip", f"{base}/archive.zip"),
		("file-zip", "Configs, results and raw data · zip", f"{base}/archive.zip?raw=1"),
	]


def _recording_row(
	project: dict, recording: dict, selected: set, progress: dict | None, failed: dict
) -> html.Tr:
	location, name = project["location"], recording["name"]
	loadable = project["error"] is None
	checked = (location, name) in selected
	opens = ["Open", icon("arrow-right", size=16)]
	return html.Tr(
		[
			html.Td(
				dmc.Checkbox(
					id={"type": "recording-check", "project": location, "index": name},
					checked=checked,
					disabled=not loadable or progress is not None,
					size="xs",
					radius="xs",
					**{"aria-label": f"Select {name}"},
				),
				className="c-check",
			),
			html.Td(html.Span(name, className="deh-mono deh-rec-name")),
			html.Td(
				[recording["window"], html.Div(recording["timezone"], className="deh-sub")]
				if loadable
				else "—"
			),
			html.Td(
				f"{recording['days']} · {recording['phases']}" if loadable else "—", className="num"
			),
			html.Td(recording["n_mice"] if loadable else "—", className="num"),
			html.Td(
				html.Div(
					[html.Span(trait, className="deh-tag") for trait in recording["traits"]],
					className="deh-tags",
				)
				if loadable
				else "—"
			),
			html.Td(recording["events"] if loadable else "—", className="num"),
			html.Td(
				_status(
					recording,
					(progress or {}).get(location, {}).get(name),
					failed.get(location, {}).get(name),
				),
				id={"type": "rec-status", "project": location, "index": name},
			),
			html.Td(
				html.Div(
					[
						components.download_menu(
							project["id"],
							name,
							html.Button(
								icon("download", size=15),
								className="deh-icon-btn sm",
								title=f"Download data of {name}",
								disabled=not loadable,
							),
						),
						dcc.Link(
							opens,
							href=dash.get_relative_path(
								f"/recording?project={project['id']}&recording={quote(name)}"
							),
							className="deh-btn deh-btn-ghost sm",
						)
						if loadable
						else html.Button(
							opens,
							className="deh-btn deh-btn-ghost sm",
							disabled=True,
							title="Opens once the configs load",
						),
						html.Button(
							icon("trash", size=15),
							id={"type": "remove-recording", "project": location, "index": name},
							className="deh-icon-btn sm deh-danger",
							title=f"Remove {name} from the project",
							disabled=not loadable or progress is not None,
						),
					],
					className="deh-card-actions",
				)
			),
		],
	)


def _project_rows(
	project: dict,
	query: str,
	expanded: list[str],
	selected: set,
	progress: dict | None,
	failed: dict,
) -> list[html.Tr]:
	location, recordings = project["location"], project["recordings"]
	visible = _visible_recordings(project, query)
	name_hit = not query or query in project["name"].lower()
	if not (name_hit or visible):
		return []

	opened = location in expanded or not name_hit
	analysed = sum(recording["done"] == recording["total"] for recording in recordings)
	partial = sum(0 < recording["done"] < recording["total"] for recording in recordings)
	share = round(100 * analysed / len(recordings)) if recordings else 0

	rows = [
		html.Tr(
			[
				html.Td(
					html.Button(
						icon("chevron-right"),
						id={"type": "project-expand", "part": "chevron", "index": location},
						className=f"deh-icon-btn deh-exp{' is-open' if opened else ''}",
						title=f"Show recordings of {project['name']}",
					),
					className="c-exp",
				),
				html.Td(
					[
						html.Div(project["name"], className="deh-p-name"),
						html.Div(project["description"], className="deh-p-desc"),
					],
					id={"type": "project-expand", "part": "name", "index": location},
					className="deh-p-cell",
				),
				html.Td(project["experimenter"]),
				html.Td(len(recordings), className="num"),
				html.Td(
					html.Div(
						[
							html.Div(
								html.I(style={"width": f"{share}%"}),
								className="deh-meter full" if share == 100 else "deh-meter",
							),
							html.Span(
								f"{analysed} of {len(recordings)} analysed"
								+ (f" · {partial} partial" if partial else "")
							),
						],
						className="deh-an",
					)
				),
				html.Td(_table_cell(project)),
				html.Td(
					html.Div(
						[
							html.Span(
								html.Bdi(location), className="deh-trunc deh-mono", title=location
							),
							html.Button(
								icon("copy", size=15),
								id={"type": "copy-path", "place": "row", "index": location},
								className="deh-icon-btn sm",
								title="Copy path",
							),
						],
						className="deh-loc",
					)
				),
				html.Td(_project_menu(project), className="c-menu"),
			],
			className="deh-project-row",
		)
	]
	rows.append(
		html.Tr(
			html.Td(
				html.Div(
					_project_detail(project, visible, selected, progress, failed) if opened else [],
					id={"type": "project-detail-body", "index": location},
					className="deh-detail-in",
				),
				colSpan=8,
			),
			id={"type": "project-detail", "index": location},
			className="deh-detail",
			hidden=not opened,
		)
	)
	return rows


def _project_detail(
	project: dict, visible: list[dict], selected: set, progress: dict | None, failed: dict
) -> list:
	"""A project's recordings table, built only while the project is open.

	Every mounted component slows each renderer update across the app, and a collapsed
	52-recording project hidden rather than left out cost seconds per click on every page.
	"""
	location, recordings = project["location"], project["recordings"]
	loadable = project["error"] is None
	keys = {(location, recording["name"]) for recording in visible}
	recordings_table = html.Table(
		[
			html.Thead(
				html.Tr(
					[
						html.Th(
							dmc.Checkbox(
								id={"type": "select-all", "index": location},
								checked=bool(keys) and keys <= selected,
								disabled=not loadable or progress is not None,
								size="xs",
								radius="xs",
								**{"aria-label": f"Select all recordings in {project['name']}"},
							),
							className="c-check",
						),
						html.Th("Recording"),
						html.Th("Window"),
						html.Th("Days · phases", className="num"),
						html.Th("Mice", className="num"),
						html.Th("Cohort"),
						html.Th("Events", className="num"),
						html.Th("Status"),
						html.Th(""),
					]
				)
			),
			html.Tbody(
				[
					_recording_row(project, recording, selected, progress, failed)
					for recording in visible
				]
				or html.Tr(
					html.Td(
						"No recordings match the filter." if recordings else "No recordings yet.",
						colSpan=9,
						className="deh-empty",
					)
				)
			),
			html.Tfoot(
				html.Tr(
					html.Td(
						html.Div(
							[
								html.Button(
									[icon("upload", size=16), "Add recordings"],
									id={
										"type": "add-recordings",
										"place": "table",
										"index": location,
									},
									className="deh-btn deh-btn-primary sm",
									disabled=not loadable,
								),
								# Shown by deh.paintSelection once two or more are selected.
								html.Button(
									[icon("trash", size=16), "Remove selected"],
									id={"type": "remove-recordings", "index": location},
									className="deh-btn sm",
									hidden=sum(key[0] == location for key in selected) < 2,
									disabled=not loadable or progress is not None,
								),
								*([_reinstate_menu(project)] if project["delisted"] else []),
							],
							className="deh-tfoot-actions",
						),
						colSpan=9,
					)
				)
			),
		],
		className="deh-tbl deh-inner",
	)
	load_error = html.Div(
		[
			icon("circle-x"),
			html.Div(
				[
					html.B("Project.load"),
					" rejects this project, so the list comes from ",
					html.Code("project.json"),
					" and the results folders. Dashboards and runs open once the configs load.",
					html.Pre(project["error"]),
				]
			),
		],
		className="deh-alert deh-alert-bad",
	)
	return [recordings_table] if loadable else [load_error, recordings_table]


@callback(
	Output("project-table", "children"),
	Output("project-count", "children"),
	Input("project-paths", "data"),
	Input("project-search", "value"),
	Input("run-failed", "data"),
	Input("data-changed", "data"),
	Input("run-active", "data"),
	# A tick repaints only the badges in flight (_render_progress); as an Input it rebuilt
	# the whole table on every finished step. Selection only repaints checkboxes, which the
	# clientside callbacks below do without a round trip.
	State("run-progress", "data"),
	State("selection", "data"),
	State("expanded", "data"),
)
def _render_projects(paths, search, failed, _written, active, progress, selection, expanded):
	# The last progress can land after the result, so it outlives the run unless gated.
	progress = progress if active else None
	paths = paths or []
	query = (search or "").strip().lower()
	selected = {tuple(key) for key in selection}
	rows = [
		*_project_rows(
			services.project_summary(location), query, expanded, selected, progress, failed
		)
		for location in paths
	]
	if not rows:
		empty = (
			f"Nothing matches “{search.strip()}”."
			if paths
			else "No projects yet. Create one, or add a project folder to see its recordings."
		)
		rows = [html.Tr(html.Td(empty, colSpan=8, className="deh-empty"))]

	head = html.Thead(
		html.Tr(
			[
				html.Th(""),
				html.Th("Project"),
				html.Th("Experimenter"),
				html.Th("Recordings", className="num"),
				html.Th("Analysis"),
				html.Th("Project table"),
				html.Th("Location"),
				html.Th(""),
			]
		)
	)
	noun = "project" if len(paths) == 1 else "projects"
	return (
		html.Table([head, html.Tbody(rows)], className="deh-tbl"),
		f"{len(paths)} {noun} remembered in this browser",
	)


@callback(
	Output("actionbar", "hidden"),
	Output("actionbar", "className"),
	Output("actionbar-message", "children"),
	Output("actionbar-meter", "style"),
	Input("selection", "data"),
	Input("run-progress", "data"),
	Input("run-active", "data"),
	State("overwrite", "checked"),
)
def _render_actionbar(selection, progress, active, overwrite):
	# The run's first report waits on its worker spawning; paint step 0 the moment it starts.
	# Here because a running background callback holds back callbacks fed by its outputs,
	# like the table's, while this one still runs.
	if active and progress is None:
		progress = services.pending_progress(selection, overwrite)
		set_props("run-progress", {"data": progress})
	if active and progress:
		jobs = [*recordings.values() for recordings in progress.values()]
		done = sum(steps for steps, _, _ in jobs)
		total = sum(steps for _, steps, _ in jobs)
		noun = "recording" if len(jobs) == 1 else "recordings"
		message = [
			icon("loader-2", class_name="deh-spin"),
			html.Span(["Analysing ", html.B(len(jobs)), f" {noun} · {done}/{total} steps"]),
		]
		return False, "deh-actionbar is-running", message, {"width": f"{100 * done // total}%"}

	noun = "recording" if len(selection) == 1 else "recordings"
	message = [html.B(len(selection)), f" {noun} selected"]
	return not selection, "deh-actionbar", message, no_update


@callback(
	Output({"type": "rec-status", "project": ALL, "index": ALL}, "children"),
	Input("run-progress", "data"),
	State("run-active", "data"),
	prevent_initial_call=True,
)
def _render_progress(progress, active):
	# Gated like _render_projects: the last tick can land after the run has ended.
	if not active or not progress:
		raise PreventUpdate
	running = [
		progress.get(output["id"]["project"], {}).get(output["id"]["index"])
		for output in ctx.outputs_list
	]
	return [no_update if job is None else _running_badge(job) for job in running]


@callback(
	Output({"type": "project-detail", "index": MATCH}, "hidden"),
	Output({"type": "project-detail-body", "index": MATCH}, "children"),
	Output({"type": "project-expand", "part": "chevron", "index": MATCH}, "className"),
	Input({"type": "project-expand", "part": ALL, "index": MATCH}, "n_clicks"),
	State({"type": "project-detail", "index": MATCH}, "hidden"),
	State("expanded", "data"),
	State("project-search", "value"),
	State("selection", "data"),
	State("run-progress", "data"),
	State("run-active", "data"),
	State("run-failed", "data"),
	prevent_initial_call=True,
)
def _toggle_project(_clicks, hidden, expanded, search, selection, progress, active, failed):
	if not ctx.triggered[0]["value"]:
		raise PreventUpdate
	location = ctx.triggered_id["index"]
	kept = [path for path in expanded if path != location]
	set_props("expanded", {"data": [*kept, location] if hidden else kept})
	chevron = f"deh-icon-btn deh-exp{' is-open' if hidden else ''}"
	if not hidden:
		return True, [], chevron

	project = services.project_summary(location)
	visible = _visible_recordings(project, (search or "").strip().lower())
	selected = {tuple(key) for key in selection}
	# Gated like _render_projects: the last progress can land after the run has ended.
	detail = _project_detail(project, visible, selected, progress if active else None, failed)
	return False, detail, chevron


clientside_callback(
	ClientsideFunction("deh", "selectRecordings"),
	Output("selection", "data", allow_duplicate=True),
	Input({"type": "recording-check", "project": ALL, "index": ALL}, "checked"),
	Input({"type": "select-all", "index": ALL}, "checked"),
	Input("selection-clear", "n_clicks"),
	State("selection", "data"),
	prevent_initial_call=True,
)

clientside_callback(
	ClientsideFunction("deh", "paintSelection"),
	Input("selection", "data"),
	State({"type": "recording-check", "project": ALL, "index": ALL}, "checked"),
	State({"type": "select-all", "index": ALL}, "checked"),
	State({"type": "remove-recordings", "index": ALL}, "hidden"),
	prevent_initial_call=True,
)


@callback(
	Output("add-modal", "opened"),
	Output("add-path", "value"),
	Output("add-path", "error"),
	Input("add-open", "n_clicks"),
	Input("add-cancel", "n_clicks"),
	Input("add-submit", "n_clicks"),
	Input("add-path", "n_submit"),
	State("add-path", "value"),
	State("project-paths", "data"),
	prevent_initial_call=True,
)
def _add_project(_open, _cancel, _submit, _enter, value, paths):
	if ctx.triggered_id == "add-open":
		return True, "", None
	if ctx.triggered_id == "add-cancel":
		return False, no_update, None

	folder = Path((value or "").strip().strip('"'))
	if folder.name == Project.MANIFEST:
		folder = folder.parent
	if not str(folder).strip(". "):
		return no_update, no_update, "Enter the folder that holds project.json."
	if not (folder.expanduser() / Project.MANIFEST).is_file():
		error = (
			f"No project.json in {folder}. Pick the folder that holds it; it sits next to one "
			"folder per recording."
		)
		return no_update, no_update, error

	location = str(folder.expanduser().resolve())
	project = services.project_summary(location)
	if location in paths:
		return no_update, no_update, f"{project['name']} is already in your list."

	set_props("project-paths", {"data": [*paths, location]})
	notify("good", f"Added {project['name']}: {len(project['recordings'])} recordings")
	return False, "", None


@callback(
	Output("create-modal", "opened"),
	Output("create-name", "value"),
	Output("create-experimenter", "value"),
	Output("create-path", "value"),
	Output("create-description", "value"),
	Output("create-name", "error"),
	Output("create-experimenter", "error"),
	Output("create-path", "error"),
	Input("create-open", "n_clicks"),
	Input("create-cancel", "n_clicks"),
	Input("create-submit", "n_clicks"),
	State("create-name", "value"),
	State("create-experimenter", "value"),
	State("create-path", "value"),
	State("create-description", "value"),
	State("project-paths", "data"),
	prevent_initial_call=True,
)
def _create_project(_open, _cancel, _submit, name, experimenter, folder, description, paths):
	unchanged = (no_update,) * 4
	if ctx.triggered_id == "create-open":
		return True, "", "", "", "", None, None, None
	if ctx.triggered_id == "create-cancel":
		return False, *unchanged, None, None, None

	name, experimenter = (name or "").strip(), (experimenter or "").strip()
	if not name or not experimenter:
		name_error = None if name else "Enter the project name."
		experimenter_error = None if experimenter else "Enter who runs the experiment."
		return no_update, *unchanged, name_error, experimenter_error, None

	location = Path((folder or "").strip().strip('"') or _PROJECTS_HOME).expanduser()
	try:
		project = Project.create(name, experimenter, location, (description or "").strip())
	except FileExistsError:
		error = "This folder already holds a project; open it with Add project."
		return no_update, *unchanged, None, None, error
	except OSError as exc:
		return no_update, *unchanged, None, None, f"Can't create this folder: {exc.strerror or exc}"

	project.close()
	set_props("project-paths", {"data": [*paths, str(project.project_location)]})
	notify("good", f"Created {name} in {project.project_location}")
	return False, *unchanged, None, None, None


for event_store, button in [
	("remove-project-event", {"type": "remove-project", "index": ALL}),
	("generate-table-event", {"type": "generate-table", "place": ALL, "index": ALL}),
	("add-recordings-event", {"type": "add-recordings", "place": ALL, "index": ALL}),
	("remove-recording-event", {"type": "remove-recording", "project": ALL, "index": ALL}),
	("remove-recordings-event", {"type": "remove-recordings", "index": ALL}),
	("reinstate-recordings-event", {"type": "reinstate-recordings", "index": ALL}),
]:
	clientside_callback(
		ClientsideFunction("deh", "clickEvent"),
		Output(event_store, "data"),
		Input(button, "n_clicks"),
		prevent_initial_call=True,
	)


@callback(
	Output("selection", "data", allow_duplicate=True),
	Input("remove-project-event", "data"),
	State("project-paths", "data"),
	State("selection", "data"),
	prevent_initial_call=True,
)
def _remove_project(event, paths, selection):
	location = event["id"]["index"]
	set_props("project-paths", {"data": [path for path in paths if path != location]})
	notify(
		"info",
		f"Removed {services.project_summary(location)['name']} from this browser's list. Its "
		"files stay on disk; add the folder again to bring it back.",
	)
	return [key for key in selection if key[0] != location]


@callback(
	Output("data-changed", "data"),
	Input("generate-table-event", "data"),
	prevent_initial_call=True,
)
def _generate_table(event):
	project = services.load_project(event["id"]["index"])
	try:
		rows = project.generate_project_table().select(pl.len()).collect().item()
	except (FileNotFoundError, ValueError) as exc:
		notify("bad", str(exc))
		return no_update
	notify("good", f"Wrote {Project.PROJECT_TABLE}: {rows:,} rows from {len(project)} recordings")
	return time.time()


def _reason(error: Exception) -> str:
	"""Why a recording was not added, worded for the person who uploaded it."""
	if isinstance(error, ValidationError):
		return "; ".join(
			(f"{'.'.join(map(str, e['loc']))}: " if e["loc"] else "")
			+ e["msg"].removeprefix("Value error, ")
			for e in error.errors()
		)
	if isinstance(error, json.JSONDecodeError):
		return f"the config file is not valid JSON ({error})"
	if isinstance(error, KeyError):
		return f"the config file has no {error} entry"
	if isinstance(error, ValueError | OSError):
		return str(error)
	return f"{type(error).__name__}: {error}"


@callback(
	Output("upload-modal", "opened"),
	Output("upload-modal", "title"),
	Output("upload-target", "data"),
	Output("upload-report", "children"),
	Output("upload-files", "contents"),
	Output("data-changed", "data", allow_duplicate=True),
	Output("replace-pending", "data"),
	Input("add-recordings-event", "data"),
	Input("upload-files", "contents"),
	Input("upload-close", "n_clicks"),
	State("upload-files", "filename"),
	State("upload-target", "data"),
	prevent_initial_call=True,
)
def _add_recordings(event, contents, _close, filenames, location):
	unchanged = (no_update,) * 7
	if ctx.triggered_id == "upload-close":
		return False, no_update, no_update, None, None, no_update, no_update
	if ctx.triggered_id == "add-recordings-event":
		location = event["id"]["index"]
		title = f"Add recordings to {services.project_summary(location)['name']}"
		return True, title, location, None, None, no_update, no_update
	if not contents:  # our own reset of the drop zone comes back through this Input
		return unchanged

	project = services.load_project(location)
	# Outlives this callback when a name is taken: the files wait there for Replace or Skip.
	# ponytail: leaks if the browser closes mid-dialog; the OS temp cleanup reclaims it.
	staging = Path(tempfile.mkdtemp(prefix="deh-upload-"))
	files = []
	for name, blob in zip(filenames, contents, strict=True):
		path = staging / Path(name).name
		path.write_bytes(base64.b64decode(blob.split(",", 1)[1]))
		files.append(path)

	report = project.add_recordings(files)
	pending = no_update
	if report.existing:
		existing = {name: [str(path) for path in paths] for name, paths in report.existing.items()}
		pending = {"location": location, "staging": str(staging), "existing": existing}
	else:
		shutil.rmtree(staging, ignore_errors=True)

	problems = [f"{source.name}: {_reason(source.error)}" for source in report.failed]
	if not problems:
		if report.added or not report.existing:
			noun = "recording" if len(report.added) == 1 else "recordings"
			notify("good", f"Added {len(report.added)} {noun} to {project.project_name}")
		return False, no_update, no_update, None, None, time.time(), pending

	more = f" (+{len(problems) - 1} more below)" if len(problems) > 1 else ""
	notify("bad" if not report.added else "warn", f"Not added - {problems[0]}{more}")
	alert = html.Div(
		[
			icon("circle-x"),
			html.Div([html.B("Not added"), html.Pre("\n".join(problems))]),
		],
		className="deh-alert deh-alert-bad",
	)
	return True, no_update, no_update, alert, None, time.time(), pending


def _replace_prompt(queue: dict) -> tuple:
	"""The Replace dialog asking about the first recording still waiting in ``queue``."""
	names = queue["waiting"]
	text = [
		html.B(names[0]),
		(
			" is already in this project. Replacing it discards its results, so it must be "
			"analysed again."
		),
	]
	more = len(names) - 1
	label = f"Do the same for the {more} other recording{'s' if more > 1 else ''} already here"
	return True, text, label, False, {"display": "flex" if more else "none"}


@callback(
	Output("replace-modal", "opened"),
	Output("replace-text", "children"),
	Output("replace-all", "label"),
	Output("replace-all", "checked"),
	Output("replace-all", "style"),
	Output("replace-queue", "data"),
	Output("data-changed", "data", allow_duplicate=True),
	Input("replace-pending", "data"),
	Input("replace-skip", "n_clicks"),
	Input("replace-confirm", "n_clicks"),
	State("replace-queue", "data"),
	State("replace-all", "checked"),
	prevent_initial_call=True,
)
def _replace_recordings(pending, _skip, _confirm, queue, apply_to_all):
	if ctx.triggered_id == "replace-pending":
		if not pending:
			raise PreventUpdate
		queue = {**pending, "waiting": list(pending["existing"]), "replace": []}
		return *_replace_prompt(queue), queue, no_update

	waiting = queue["waiting"]
	decided = len(waiting) if apply_to_all else 1
	if ctx.triggered_id == "replace-confirm":
		queue["replace"] += waiting[:decided]
	queue["waiting"] = waiting[decided:]
	if queue["waiting"]:
		return *_replace_prompt(queue), queue, no_update

	# Every name is decided: add the ones to replace, then let go of the staged files.
	project = services.load_project(queue["location"])
	files = [*queue["existing"][name] for name in queue["replace"]]
	report = project.add_recordings(files, overwrite=True) if files else None
	shutil.rmtree(queue["staging"], ignore_errors=True)

	if report is None:
		notify("info", "Nothing replaced; the recordings already in the project stay as they were")
		return False, no_update, no_update, no_update, no_update, None, no_update
	messages = []
	if report.replaced:
		messages.append(
			f"Replaced {', '.join(report.replaced)} in {project.project_name}; "
			"the previous results were discarded"
		)
	if report.failed:
		reasons = "; ".join(f"{f.name}: {_reason(f.error)}" for f in report.failed)
		messages.append(f"Not replaced - {reasons}")
	notify("bad" if not report.replaced else "warn", ". ".join(messages))
	return False, no_update, no_update, no_update, no_update, None, time.time()


@callback(
	Output("remove-modal", "opened"),
	Output("remove-modal", "title"),
	Output("remove-target", "data"),
	Output("data-changed", "data", allow_duplicate=True),
	Output("selection", "data", allow_duplicate=True),
	Input("remove-recording-event", "data"),
	Input("remove-recordings-event", "data"),
	Input("remove-cancel", "n_clicks"),
	Input("remove-delist", "n_clicks"),
	Input("remove-delete", "n_clicks"),
	State("remove-target", "data"),
	State("selection", "data"),
	prevent_initial_call=True,
)
def _remove_recording(event, bulk_event, _cancel, _delist, _delete, target, selection):
	if ctx.triggered_id in ("remove-recording-event", "remove-recordings-event"):
		if ctx.triggered_id == "remove-recording-event":
			location, names = event["id"]["project"], [event["id"]["index"]]
		else:
			location = bulk_event["id"]["index"]
			names = [name for project, name in selection if project == location]
		title = f"Remove {names[0]}?" if len(names) == 1 else f"Remove {len(names)} recordings?"
		return True, title, [location, names], no_update, no_update
	if ctx.triggered_id == "remove-cancel":
		return False, no_update, no_update, no_update, no_update

	location, names = target
	delete_files = ctx.triggered_id == "remove-delete"
	project = services.load_project(location)
	for name in names:
		project.remove_recording(name, delete_files=delete_files)
	what = names[0] if len(names) == 1 else f"{len(names)} recordings"
	notify(
		"info",
		f"Deleted {what} and the files" if delete_files else f"Delisted {what}; the files stay",
	)
	kept = [key for key in selection if key[0] != location or key[1] not in names]
	return False, no_update, no_update, time.time(), kept


clientside_callback(
	ClientsideFunction("deh", "selectDelisted"),
	Output({"type": "reinstate-pick", "index": MATCH}, "value"),
	Output({"type": "reinstate-all", "index": MATCH}, "checked"),
	Output({"type": "reinstate-all", "index": MATCH}, "indeterminate"),
	Input({"type": "reinstate-all", "index": MATCH}, "checked"),
	Input({"type": "reinstate-pick", "index": MATCH}, "value"),
	State({"type": "reinstate-names", "index": MATCH}, "data"),
	prevent_initial_call=True,
)


@callback(
	Output("data-changed", "data", allow_duplicate=True),
	Input("reinstate-recordings-event", "data"),
	State({"type": "reinstate-pick", "index": ALL}, "value"),
	prevent_initial_call=True,
)
def _reinstate_recordings(event, _picks):
	location = event["id"]["index"]
	names = next(pick["value"] for pick in ctx.states_list[0] if pick["id"]["index"] == location)
	if not names:
		raise PreventUpdate
	project = services.load_project(location)
	reinstated = []
	for name in names:
		try:
			project.reinstate_recording(name)
		except FileNotFoundError as exc:
			notify("bad", str(exc))
		else:
			reinstated.append(name)
	if not reinstated:
		return no_update
	what = reinstated[0] if len(reinstated) == 1 else f"{len(reinstated)} recordings"
	notify("good", f"Reinstated {what}")
	return time.time()


@callback(
	Output("params-ranking-status", "children"),
	Output("params-ranking-upload", "contents"),
	Input("params-open", "n_clicks"),
	Input("params-ranking-upload", "contents"),
	Input("params-ranking-clear", "n_clicks"),
	State("params-ranking-upload", "filename"),
	State("selection", "data"),
	prevent_initial_call=True,
)
def _prev_rankings(_open, contents, _clear, filenames, selection):
	uploaded = ctx.triggered_id == "params-ranking-upload"
	if uploaded and not contents:  # our own reset of the drop zone comes back through this Input
		return no_update, no_update

	recordings = [services.load_project(location)[name] for location, name in selection]
	if ctx.triggered_id == "params-ranking-clear":
		for recording in recordings:
			recording.set_prev_ranking(None)
	elif uploaded:
		problems = []
		for filename, blob in zip(filenames, contents, strict=True):
			stem = Path(filename).stem
			targets = [recording for recording in recordings if recording.name == stem]
			if not targets:
				problems.append(f"{filename}: no selected recording is named {stem}.")
				continue

			raw = base64.b64decode(blob.split(",", 1)[1])
			try:
				frame = (
					pl.read_csv(io.BytesIO(raw), schema_overrides={"animal_id": pl.String})
					if filename.lower().endswith(".csv")
					else pl.read_parquet(io.BytesIO(raw))
				)
				for recording in targets:
					recording.set_prev_ranking(frame)
			except Exception as exc:
				problems.append(f"{filename}: {exc}")
		if problems:
			notify("warn", " ".join(problems))

	status = []
	for recording in recordings:
		if (ratings := recording.prev_ranking) is not None:
			status += [html.Code(recording.name), f" · {ratings.height} animals", html.Br()]
	status = html.Span(status[:-1] or "None stored for the selection")
	return status, None if uploaded else no_update


@callback(
	Output("params-drawer", "opened"),
	Output("analysis-params", "data"),
	Output("params-minimum-time", "error"),
	Output("params-minimum-alone", "error"),
	Input("params-open", "n_clicks"),
	Input("params-apply", "n_clicks"),
	State("params-minimum-time", "value"),
	State("params-minimum-alone", "value"),
	State("params-chasing", "value"),
	prevent_initial_call=True,
)
def _apply_params(_open, _apply, minimum_time, minimum_time_alone, chasing_time_window):
	if ctx.triggered_id == "params-open":
		return True, no_update, None, None

	errors = [
		None if isinstance(value, int | float) else "A number of seconds, 0 or more."
		for value in (minimum_time, minimum_time_alone)
	]
	if any(errors):
		return no_update, no_update, *errors

	params = {
		"minimum_time": minimum_time,
		"minimum_time_alone": minimum_time_alone,
		"chasing_time_window": chasing_time_window,
	}
	notify("info", "Parameters apply to the next run.")
	return False, params, None, None


clientside_callback(
	ClientsideFunction("deh", "resetParams"),
	Output("params-minimum-time", "value"),
	Output("params-minimum-alone", "value"),
	Output("params-chasing", "value"),
	Input("params-reset", "n_clicks"),
	State("params-defaults", "data"),
	prevent_initial_call=True,
)


callback(
	Output("run-failed", "data"),
	Output("selection", "data", allow_duplicate=True),
	Input("run-start", "n_clicks"),
	State("selection", "data"),
	State("analysis-params", "data"),
	State("overwrite", "checked"),
	State("use-prev-ranking", "checked"),
	State("run-failed", "data"),
	background=True,
	progress=Output("run-progress", "data"),
	running=[
		(Output("run-start", "disabled"), True, False),
		(Output("run-active", "data"), True, False),
	],
	prevent_initial_call=True,
)(services.run_analysis)

clientside_callback(
	ClientsideFunction("deh", "resetProgress"),
	Output("run-progress", "data", allow_duplicate=True),
	Input("run-start", "n_clicks"),
	prevent_initial_call=True,
)


@callback(Input("run-cancel", "n_clicks"), State("selection", "data"), prevent_initial_call=True)
def _cancel_run(_clicks, selection):
	for location in {location for location, _ in selection}:
		services.request_cancel(location)
	notify("info", "Cancelling: recordings in flight finish their current step first.")


clientside_callback(
	ClientsideFunction("deh", "copyPath"),
	Input({"type": "copy-path", "place": ALL, "index": ALL}, "n_clicks"),
	prevent_initial_call=True,
)
