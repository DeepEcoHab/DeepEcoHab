import hashlib
import json
import threading
import time
import urllib.request
from collections.abc import Callable
from functools import cache, lru_cache
from importlib.metadata import version
from pathlib import Path
from typing import Any

import diskcache
import polars as pl
import polars.selectors as cs
from dash import no_update
lazy from packaging.version import Version

from deepecohab import AnalysisParams, Project, Recording, recording_status
from deepecohab.app.builder import catalog as builder_catalog
from deepecohab.app.components import notify
from deepecohab.core.data_model import DataFrameRegistry, Layout
from deepecohab.plotting import PlotContext, durations

BUILDER_PRESETS_FILE = "builder_presets.json"

CACHE_DIR = Path.home() / ".deepecohab" / "app-cache"


def project_id(location: str) -> str:
	"""A short, stable id for a project folder, so URLs never carry its path."""
	return hashlib.blake2b(location.encode(), digest_size=6).hexdigest()


def resolve_project(paths: list[str] | None, pid: str) -> str | None:
	"""The remembered project folder whose id is ``pid``, or None if it is not among ``paths``."""
	return next((path for path in paths or [] if project_id(path) == pid), None)


def remember_project(location: str) -> str:
	"""This project's id, and cache the id -> path mapping for the download routes.

	The browser's project list lives in local storage, which a plain Flask route
	cannot read, so every time a project is summarised for the Projects table this
	registers it; a download link's route then looks the path back up by id. Written
	once per process: opening the disk cache costs more than the rest of the summary.
	"""
	return _remember(location, CACHE_DIR)


@cache
def _remember(location: str, cache_dir: Path) -> str:
	# cache_dir is part of the cache key, not just the destination: a test that points
	# CACHE_DIR at a tmp_path would otherwise get the previous test's hit and write nothing.
	pid = project_id(location)
	with diskcache.Cache(cache_dir) as disk:
		disk.set(f"path:{pid}", location)
	return pid


def resolve_project_path(pid: str) -> str | None:
	"""The project folder last registered under ``pid`` by :func:`remember_project`."""
	with diskcache.Cache(CACHE_DIR) as disk:
		return disk.get(f"path:{pid}")


def _manifest(root: Path) -> dict:
	"""The project's ``project.json``, or an empty dict when it cannot be read."""
	try:
		return json.loads((root / Project.MANIFEST).read_text(encoding="utf-8"))
	except OSError, ValueError:
		return {}


def project_name(location: str) -> str:
	"""The name in the project's manifest, or its folder name when that cannot be read."""
	return _manifest(Path(location)).get("project_name", Path(location).name)


def csv_ready(frame: pl.DataFrame) -> pl.DataFrame:
	"""``frame`` with every Duration column in seconds and every Enum/Categorical as a string.

	CSV has neither type, so a plain ``write_csv`` would otherwise dump raw microsecond
	integers for a duration and silently drop a category's declared order.
	"""
	return frame.with_columns(
		cs.duration().dt.total_microseconds() / 1_000_000,
		(cs.enum() | cs.categorical()).cast(pl.String),
	)


def load_project(location: str) -> Project:
	"""The project at ``location``, loaded again only once its manifest or a config changes."""
	root = Path(location)
	configs = sorted(root.glob(f"*/{Project.CONFIG}"))
	stamp = tuple(path.stat().st_mtime_ns for path in [root / Project.MANIFEST, *configs])
	return _load_project(location, stamp)


@lru_cache(maxsize=16)
def _load_project(location: str, _stamp: tuple[int, ...]) -> Project:
	return Project.load(Path(location))


def project_summary(location: str) -> dict:
	"""Everything the Projects table shows for one folder.

	A project that ``Project.load`` rejects still lists its recordings, from ``project.json``
	and each recording's results folder, next to the error that stopped it.

	Args:
		location: the project folder, as remembered in the browser.

	Returns:
		The manifest fields, ``error`` (None when the project loads), ``table_rows`` (None
		until the project table is generated), one dict per recording and the ``delisted``
		names. A recording's window, cohort and events are there only when the project loads.
	"""
	root = Path(location)
	steps = len(DataFrameRegistry.step_order())
	table = root / Project.PROJECT_TABLE
	summary = {
		"location": location,
		"id": remember_project(location),
		"table_rows": (
			pl.scan_parquet(table).select(pl.len()).collect().item() if table.is_file() else None
		),
	}

	try:
		project = load_project(location)
	except Exception as exc:  # any unreadable folder still gets its row, with the reason
		manifest = _manifest(root)
		return {
			**summary,
			"name": manifest.get("project_name", root.name),
			"experimenter": manifest.get("experimenter", ""),
			"description": manifest.get("description", ""),
			"error": f"{type(exc).__name__}: {exc}",
			"recordings": [
				{"name": name, "done": sum(recording_status(root / name).values()), "total": steps}
				for name in manifest.get("data_catalog", {})
			],
			"delisted": list(manifest.get("delisted", {})),
			"table_stale": False,
		}

	return {
		**summary,
		"name": project.project_name,
		"experimenter": project.experimenter,
		"description": project.description,
		"error": None,
		"recordings": [_recording_summary(recording, steps) for recording in project.recordings],
		"delisted": list(project.delisted),
		"table_stale": table.is_file() and _table_stale(project, table),
	}


def _table_stale(project: Project, table: Path) -> bool:
	"""Whether the project table misses a listed recording or predates any of its results."""
	built = table.stat().st_mtime_ns
	listed = set(pl.scan_parquet(table).select(pl.col("recording").unique()).collect().to_series())
	return listed != {recording.name for recording in project.recordings} or any(
		path.stat().st_mtime_ns > built
		for recording in project.recordings
		for path in recording.results_path.glob("*.parquet")
	)


def _recording_summary(recording: Recording, steps: int) -> dict:
	timeline = recording.timeline
	zone = timeline.recording_timezone
	start = timeline.start_datetime.astimezone(zone)
	end = timeline.end_datetime.astimezone(zone)
	animals = recording.cohort.animals
	return {
		"name": recording.name,
		"done": sum(recording_status(recording.root).values()),
		"total": steps,
		"window": f"{start.day} {start:%b} → {end.day} {end:%b %Y}",
		"timezone": str(zone),
		"days": timeline.days_range[1] - timeline.days_range[0] + 1,
		"phases": timeline.phase_range[1] - timeline.phase_range[0] + 1,
		"n_mice": recording.cohort.n_mice,
		"traits": [
			*dict.fromkeys(getattr(animal, field) for animal in animals)
			for field in ("mouse_line", "genotype", "sex")
		],
		"events": len(recording.events),
	}


def plot_context(location: str, name: str) -> PlotContext:
	"""The plot context for one recording, rebuilt only once its results change."""
	return _plot_context(location, name, results_stamp(location, name))


def results_stamp(location: str, name: str) -> tuple[tuple[str, int], ...]:
	"""Each results table's name and mtime, so it changes whenever the recording is re-analysed."""
	results = load_project(location)[name].results_path
	if not results.is_dir():
		return ()
	return tuple(sorted((path.name, path.stat().st_mtime_ns) for path in results.glob("*.parquet")))


# Two: a context holds whole result tables in memory, which for a long recording runs to
# gigabytes, and a table read back from disk costs tens of milliseconds.
@lru_cache(maxsize=2)
def _plot_context(location: str, name: str, _stamp: tuple[tuple[str, int], ...]) -> PlotContext:
	return PlotContext.from_recording(load_project(location)[name])


#: Driven by the control bar rather than a per-card control: builder option -> rec-controls key.
GLOBAL_OPTIONS = frozendict(
	{
		"days_range": "window",
		"granularity": "granularity",
		"phase_type": "phases",
		"color_by": "color_by",
		"label_by": "label_by",
		"hours_range": "hours",
		"group_mean": "group_mean",
	}
)


def plot_options(request: dict) -> tuple[PlotContext, dict[str, Any]]:
	"""The context and builder options a recording card's plot request names."""
	context_data, controls = request["context"], request["controls"]
	context = plot_context(context_data["location"], context_data["recording"])
	# A cleared NumberInput sends "", which falls back to the builder's default.
	values = {key: value for key, value in request["opts"].items() if value != ""}
	# deh.plotRequest sends only the controls this plot takes.
	values.update(
		(option, controls[key]) for option, key in GLOBAL_OPTIONS.items() if key in controls
	)
	# None for the untouched full day, matching every builder's own default: some prepare.*
	# steps aggregate away the hour column before applying this filter and only skip that
	# filter when it is exactly None (e.g. prep_polar).
	if values.get("hours_range") == [0, 23]:
		values["hours_range"] = None
	return context, values


def recording_summary(location: str, name: str) -> dict:
	"""Everything the recording dashboard header and control bar show."""
	recording = load_project(location)[name]
	timeline = recording.timeline
	zone = timeline.recording_timezone
	context = plot_context(location, name)

	return {
		"name": recording.name,
		"project": recording.project_name,
		"timezone": str(zone),
		"start": timeline.start_datetime.astimezone(zone).isoformat(),
		"end": timeline.end_datetime.astimezone(zone).isoformat(),
		"days": timeline.days_range[1] - timeline.days_range[0] + 1,
		"phases": timeline.phase_range[1] - timeline.phase_range[0] + 1,
		"start_from": timeline.start_from,
		"onsets": {phase: onset.isoformat("minutes") for phase, onset in timeline.phases.items()},
		"location": recording.recording_location,
		"n_mice": recording.cohort.n_mice,
		"cages": len(recording.layout.cages),
		"tunnels": len(recording.layout.tunnels),
		"events": [event.name for event in recording.events],
		"notes": recording.notes,
		"quality": quality_summary(context) if "recording_quality" in context else None,
	}


def update_notes(location: str, name: str, notes: str, tag: str | None = None) -> None:
	"""Persist new notes for one recording or animal; the next `load_project` picks them up."""
	load_project(location).update_notes(name, notes, tag)


def quality_summary(context: PlotContext) -> dict:
	"""Pooled detection-quality stats for the header and the quality tab's tiles."""
	frame = context.table("recording_quality").with_columns(pl.col("animal_id").cast(pl.String))
	counts = pl.col("correct", "interpolated", "bad").sum()
	missed = pl.col("interpolated") + pl.col("bad")
	passes = pl.col("correct") + missed
	rate = pl.when(passes > 0).then(100 * missed / passes).otherwise(0.0).alias("miss")

	pooled = frame.select(counts).with_columns(rate).row(0, named=True)

	return {
		**pooled,
		"antenna_stats": {
			str(row.pop("antenna")): row
			for row in frame.group_by("antenna")
			.agg(counts)
			.with_columns(rate)
			.iter_rows(named=True)
		},
	}


def reads_by_animal(context: PlotContext) -> dict[str, dict[str, int]]:
	"""Each animal's ``correct``, ``interpolated`` and ``bad`` passes over every antenna."""
	return {
		row.pop("animal_id"): row
		for row in context.table("recording_quality")
		.group_by(pl.col("animal_id").cast(pl.String))
		.agg(pl.col("correct", "interpolated", "bad").sum())
		.iter_rows(named=True)
	}


def overview_tiles(context: PlotContext) -> list[dict]:
	"""What the recording amounts to, for the Overview tab's stat card.

	Every tile that needs a table the pipeline has not written is left out rather than
	blanked, so a half-analysed recording still shows what it does have.

	Returns:
		One ``label`` / ``value`` / ``note`` dict per tile, in reading order.
	"""
	assert context.recording is not None
	days = context.days_range[1] - context.days_range[0] + 1
	phases = context.phase_range[1] - context.phase_range[0] + 1
	tiles = [
		{
			"label": "Duration",
			"value": f"{days} d",
			"note": f"{phases} phases · {len(context.animal_ids)} mice",
		}
	]

	if "main_df" in context:
		# Lazily: this card is built for every recording that is opened, and main_df runs
		# to hundreds of thousands of rows that nothing here reads except to count them.
		rows = context.recording.load_results("main_df", eager=False).select(pl.len())
		note = ""
		if "activity_df" in context:
			visits = _placed(context).select(pl.sum("visits_to_position")).item()
			note = f"{visits:,} visits".replace(",", " ")
		tiles.append(
			{
				"label": "Registrations",
				"value": f"{rows.collect().item():,}".replace(",", " "),
				"note": note,
			}
		)

	if "chasings_df" in context:
		frame = context.table("chasings_df")
		won = frame.group_by("chaser").agg(pl.sum("chasings")).sort("chasings", descending=True)
		total = int(frame["chasings"].sum())
		tiles.append(
			{
				"label": "Chasings",
				"value": f"{total:,}".replace(",", " "),
				"note": f"top animal won {won['chasings'][0]} - {won['chaser'][0]}"
				if total
				else "none recorded",
			}
		)

	if "activity_df" in context and context.cages:
		share = (
			_placed(context)
			.filter(pl.col("position").is_in(context.cages))
			.group_by("position")
			.agg(pl.col("time_in_position").sum().dt.total_seconds().alias("seconds"))
			.with_columns((100 * pl.col("seconds") / pl.col("seconds").sum()).alias("pct"))
			.sort("pct", descending=True)
			.row(0, named=True)
		)
		kind = next(
			cage.cage_type
			for cage in context.recording.layout.cages
			if cage.name == share["position"]
		)
		tiles.append(
			{
				"label": "Busiest cage",
				"value": share["position"].capitalize().replace("_", " "),
				"note": f"{share['pct']:.0f}% of cohort time · {kind}",
			}
		)

	return tiles


def _placed(context: PlotContext) -> pl.DataFrame:
	"""``activity_df`` without the undefined sentinel, which is not a place."""
	return context.table("activity_df").filter(
		pl.col("position").cast(pl.String) != Layout.UNDEFINED
	)


def missing_time(context: PlotContext) -> dict:
	"""Time each animal spent at a position antennas could not resolve, and its share.

	Args:
		context: the plot context holding ``activity_df`` and ``phase_durations``.

	Returns:
		``rows`` (worst first): ``animal_id``, ``time_in_position_text`` and ``share`` as
		a percentage of the recording length, plus the cohort ``mean_share``.
	"""
	# Series.sum() is typed as a broad scalar union; it is a Duration here, same column.
	length = context.table("phase_durations")["duration"].sum().total_seconds()  # ty: ignore[unresolved-attribute]

	frame = (
		context.table("activity_df")
		.with_columns(pl.col("animal_id").cast(pl.String))
		.filter(pl.col("position").cast(pl.String) == Layout.UNDEFINED)
		.group_by("animal_id")
		.agg(pl.col("time_in_position").sum())
		.join(pl.DataFrame({"animal_id": context.animal_ids}), on="animal_id", how="right")
		.with_columns(pl.col("time_in_position").fill_null(pl.duration(microseconds=0)))
		.with_columns(
			(100 * pl.col("time_in_position").dt.total_seconds(fractional=True) / length).alias(
				"share"
			)
		)
		.sort("share", descending=True)
	)
	frame, _label = durations.to_display(frame, "time_in_position", "auto", "Time unknown")

	return {
		"rows": frame.select("animal_id", "time_in_position_text", "share").to_dicts(),
		# Series.mean() is typed as a broad scalar union; "share" is a plain float column.
		"mean_share": float(frame["share"].mean()),  # ty: ignore[invalid-argument-type]
	}


def event_recording_counts(project: Project) -> dict[str, int]:
	"""How many recordings declare each event, "Any event" counting recordings with any."""
	counts = dict.fromkeys(project.event_names(), 0)
	for recording in project.recordings:
		declared = {event.name for event in recording.events}
		for name in declared:
			counts[name] += 1
		if declared:
			counts["Any event"] += 1
	return counts


def builder_frame(location: str) -> tuple[pl.LazyFrame, list[builder_catalog.Field]]:
	"""The prepared project table and its field catalog for the plot builder.

	Raises:
		FileNotFoundError: the project table has not been generated yet.
	"""
	project = load_project(location)
	return builder_catalog.prepare(project.load_project_table(), project.event_names())


def load_saved_presets(location: str) -> list[dict]:
	"""Builder presets saved to ``<project>/builder_presets.json``, empty if there are none."""
	path = Path(location) / BUILDER_PRESETS_FILE
	if not path.is_file():
		return []
	return json.loads(path.read_text(encoding="utf-8"))


def _write_presets(location: str, presets: list[dict]) -> None:
	"""Overwrite ``<project>/builder_presets.json``."""
	path = Path(location) / BUILDER_PRESETS_FILE
	path.write_text(json.dumps(presets, indent=2), encoding="utf-8")


def save_preset(location: str, preset: dict) -> None:
	"""Add or replace one preset in ``<project>/builder_presets.json``."""
	kept = [p for p in load_saved_presets(location) if p["id"] != preset["id"]]
	_write_presets(location, [*kept, preset])


def delete_saved_preset(location: str, preset_id: str) -> None:
	"""Remove one preset from ``<project>/builder_presets.json``."""
	_write_presets(location, [p for p in load_saved_presets(location) if p["id"] != preset_id])


def request_cancel(location: str) -> None:
	"""Ask a running analysis of ``location`` to stop once its steps in flight land."""
	with diskcache.Cache(CACHE_DIR) as flags:
		flags.set(f"cancel:{location}", True, expire=3600)


def run_analysis(
	set_progress: Callable[[dict], None],
	_clicks: int,
	selection: list[list[str]],
	params: dict | None,
	overwrite: bool,
	use_prev_ranking: bool,
	failed: dict[str, dict[str, str]],
) -> tuple:
	"""The Run analysis background callback: the selected recordings, project by project.

	Progress goes out as ``{location: {recording: [steps done, steps, step building]}}``.
	A watcher thread turns :func:`request_cancel` into the pipeline's cancel event, so the
	steps in flight still land. Recordings left incomplete by a project that raised are
	marked failed.

	The stored previous rankings are only optional when overwriting: that switch is hidden
	otherwise, so a ranking built for the first time always starts from them.

	Returns:
		The failed-recordings store and the selection, cleared after a clean run. The outcome
		goes out as a notification.
	"""
	params = {**(params or {}), "use_prev_ranking": use_prev_ranking or not overwrite}

	order = DataFrameRegistry.step_order()
	progress = pending_progress(selection, overwrite)
	todo = {location: list(recordings) for location, recordings in progress.items()}

	if not todo:
		message = (
			f"Nothing to build: the selected recordings already have all {len(order)} tables. "
			"Turn on Overwrite to rebuild them."
		)
		notify("warn", message)
		return no_update, no_update

	started = time.perf_counter()
	cancel = threading.Event()
	errors = _analyse_projects(
		progress, AnalysisParams(**params), overwrite, order, set_progress, cancel
	)

	failed = dict(failed or {})
	for location, names in todo.items():
		kept = {
			name: reason for name, reason in failed.get(location, {}).items() if name not in names
		}
		if location in errors:
			kept |= {name: str(errors[location]) for name in names if not _analysed(location, name)}
		failed[location] = kept

	if errors:
		notify("bad", f"Analysis failed: {next(iter(errors.values()))}")
		return failed, no_update
	if cancel.is_set():
		notify(
			"warn",
			"Cancelled. Recordings in flight finished their current step first, and their "
			"finished tables stay on disk.",
		)
		return failed, no_update

	count = sum(map(len, todo.values()))
	noun = "recording" if count == 1 else "recordings"
	notify("good", f"Analysed {count} {noun} in {time.perf_counter() - started:.1f} s")
	return failed, []


def pending_progress(selection: list[list[str]], overwrite: bool) -> dict:
	"""The progress a run starts from: each selected recording it will build, at step 0.

	The click paints it straight away; the background run only reports once its worker is up.
	"""
	order = DataFrameRegistry.step_order()
	progress: dict[str, dict[str, list]] = {}
	for location, name in selection:
		if overwrite or not _analysed(location, name):
			progress.setdefault(location, {})[name] = [0, len(order), order[0]]
	return progress


def _analyse_projects(
	progress: dict,
	params: AnalysisParams,
	overwrite: bool,
	order: list[str],
	set_progress: Callable[[dict], None],
	cancel: threading.Event,
) -> dict[str, Exception]:
	"""Analyse each project in turn, reporting progress and collecting the ones that raised.

	``progress`` is :func:`pending_progress`, updated in place as steps land. A watcher
	thread turns :func:`request_cancel` into ``cancel``, so the steps in flight still land.
	``cancel`` is the caller's, so it can tell a cancelled run from a clean one.
	"""
	todo = {location: list(recordings) for location, recordings in progress.items()}
	errors: dict[str, Exception] = {}
	finished = threading.Event()

	with diskcache.Cache(CACHE_DIR) as flags:
		keys = [f"cancel:{location}" for location in todo]
		for key in keys:
			flags.delete(key)
		# Cancel shows once progress does, so a click can no longer land before this reset.
		set_progress(progress)

		def watch() -> None:
			while not finished.wait(0.5):
				if any(flags.get(key) for key in keys):
					cancel.set()

		def analyse(location: str, names: list[str]) -> None:
			with Project.load(Path(location)) as project:
				for event in project.iter_analysis(
					params, names, overwrite=overwrite, cancel=cancel
				):
					building = order[event.step_index] if event.step_index < len(order) else ""
					progress[location][event.recording] = [event.step_index, len(order), building]
					set_progress(progress)

		threading.Thread(target=watch, daemon=True).start()
		try:
			for location, names in todo.items():
				if cancel.is_set():
					break
				try:
					analyse(location, names)
				except Exception as exc:  # shown to the user; the other projects still run
					errors[location] = exc
		finally:
			finished.set()

	return errors


def _analysed(location: str, name: str) -> bool:
	return all(recording_status(Path(location) / name).values())


# ponytail: local installs only - delete with its _check_update and _update callbacks,
# updater.py and the toast's listener in clientside.js once the app is deployed and users
# no longer update it.
@cache
def newer_release() -> str | None:
	"""The latest stable PyPI release if it is newer than the installed one, else None.

	PyPI's ``info.version`` skips pre-releases, so an installed rc above the last stable
	release compares as newer and nothing is offered. None too when PyPI is unreachable.
	"""
	try:
		with urllib.request.urlopen("https://pypi.org/pypi/deepecohab/json", timeout=3) as r:
			latest = json.load(r)["info"]["version"]
	except OSError, ValueError, KeyError:
		return None
	return latest if Version(latest) > Version(version("deepecohab")) else None
