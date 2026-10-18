import hashlib
import json
import urllib.request
import uuid
from functools import cache, lru_cache
from importlib.metadata import version
from pathlib import Path
from typing import Any

import diskcache
import polars as pl
from dash import no_update
lazy from packaging.version import Version

from deepecohab import AnalysisParams, Project, Recording, recording_status
from deepecohab.app.builder import catalog as builder_catalog
from deepecohab.app.components import notify
from deepecohab.core.data_model import AnalysisRun, DataFrameRegistry, atomic_write_text
from deepecohab.plotting import PlotContext, durations, summaries

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
	"""The project's manifest, or an empty dict when it cannot be read."""
	try:
		return Project.read_manifest(root)
	except OSError, ValueError:
		return {}


def project_name(location: str) -> str:
	"""The name in the project's manifest, or its folder name when that cannot be read."""
	return _manifest(Path(location)).get("project_name", Path(location).name)


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
		"table_stale": project.project_table_stale(),
	}


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
		# Series.mean() is typed as a broad scalar union; "share" is a plain float column.
		"unknown": float(summaries.unknown_time(context)["share"].mean())  # ty: ignore[invalid-argument-type]
		if "activity_df" in context and "phase_durations" in context
		else None,
	}


def update_notes(location: str, name: str, notes: str, tag: str | None = None) -> None:
	"""Persist new notes for one recording or animal; the next `load_project` picks them up."""
	load_project(location).update_notes(name, notes, tag)


def overview_tiles(context: PlotContext) -> list[dict]:
	"""What the recording amounts to, for the Overview tab's stat card.

	Every tile that needs a table the pipeline has not written is left out rather than
	blanked, so a half-analysed recording still shows what it does have.

	Returns:
		One ``label`` / ``value`` / ``note`` dict per tile, in reading order.
	"""
	overview = summaries.recording_overview(context)
	tiles = [
		{
			"label": "Duration",
			"value": f"{overview['days']} d",
			"note": f"{overview['phases']} phases · {overview['mice']} mice",
		}
	]

	if "registrations" in overview:
		visits = overview.get("visits")
		tiles.append(
			{
				"label": "Registrations",
				"value": _spaced(overview["registrations"]),
				"note": "" if visits is None else f"{_spaced(visits)} visits",
			}
		)

	if "chasings" in overview:
		top = overview["top_chaser"]
		tiles.append(
			{
				"label": "Chasings",
				"value": _spaced(overview["chasings"]),
				"note": f"top animal won {top['chasings']} - {top['chaser']}"
				if top
				else "none recorded",
			}
		)

	if cage := overview.get("busiest_cage"):
		tiles.append(
			{
				"label": "Busiest cage",
				"value": cage["position"].capitalize().replace("_", " "),
				"note": f"{cage['pct']:.0f}% of cohort time · {cage['cage_type']}",
			}
		)

	return tiles


def _spaced(count: int) -> str:
	return f"{count:,}".replace(",", " ")


def missing_time(context: PlotContext) -> dict:
	"""Time each animal spent at a position antennas could not resolve, and its share.

	Returns:
		``rows`` (worst first): ``animal_id``, ``time_in_position_text`` and ``share`` as
		a percentage of the recording length, plus the cohort ``mean_share``.
	"""
	frame = summaries.unknown_time(context)
	frame, _label = durations.to_display(frame, "time_in_position", "auto", "Time unknown")
	return {
		"rows": frame.select("animal_id", "time_in_position_text", "share").to_dicts(),
		# Series.mean() is typed as a broad scalar union; "share" is a plain float column.
		"mean_share": float(frame["share"].mean()),  # ty: ignore[invalid-argument-type]
	}


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
	atomic_write_text(Path(location) / BUILDER_PRESETS_FILE, json.dumps(presets, indent=2))


def save_preset(location: str, preset: dict) -> None:
	"""Add or replace one preset in ``<project>/builder_presets.json``."""
	kept = [p for p in load_saved_presets(location) if p["id"] != preset["id"]]
	_write_presets(location, [*kept, preset])


def delete_saved_preset(location: str, preset_id: str) -> None:
	"""Remove one preset from ``<project>/builder_presets.json``."""
	_write_presets(location, [p for p in load_saved_presets(location) if p["id"] != preset_id])


# ponytail: runs live in this process's memory, so the app must be served by one process;
# a multi-worker deploy needs them in shared storage, or behind the API's own run table.
_runs: dict[str, tuple[str, AnalysisRun]] = {}


def start_run(
	selection: list[list[str]], params: dict | None, overwrite: bool, use_prev_ranking: bool
) -> str | None:
	"""Start analysing the selected recordings that need it; the run's id, None if none do.

	The selection is one project's: the Projects page cannot select across projects.
	The stored previous rankings are only optional when overwriting: that switch is hidden
	otherwise, so a ranking built for the first time always starts from them.
	"""
	location = selection[0][0]
	names = [name for _, name in selection]
	if not overwrite:
		names = load_project(location).pending(names)
	if not names:
		steps = len(DataFrameRegistry.step_order())
		notify(
			"warn",
			f"Nothing to build: the selected recordings already have all {steps} tables. "
			"Turn on Overwrite to rebuild them.",
		)
		return None

	params = {**(params or {}), "use_prev_ranking": use_prev_ranking or not overwrite}
	run = AnalysisRun(Path(location), names, AnalysisParams(**params), overwrite=overwrite)
	run_id = uuid.uuid4().hex
	_runs[run_id] = location, run
	return run_id


def run_progress(run_id: str) -> dict:
	"""``{location: {recording: [steps done, steps]}}`` of a run."""
	location, run = _runs[run_id]
	return {location: run.progress()}


def cancel_run(run_id: str) -> None:
	"""Ask a run to stop once its steps in flight land."""
	if run_id in _runs:
		_runs[run_id][1].cancel()


def analysis_running() -> bool:
	"""Whether any run in this process is still building."""
	return any(not run.done for _, run in _runs.values())


def finish_run(run_id: str, failed: dict[str, dict[str, str]] | None) -> tuple | None:
	"""None while the run goes on; then the failed-recordings store and the selection.

	Recordings a failed run left incomplete are marked failed, and the selection is cleared
	after a clean run. The outcome goes out as a notification.
	"""
	location, run = _runs[run_id]
	if not run.done:
		return None
	del _runs[run_id]

	names = list(run.progress())
	failed = dict(failed or {})
	kept = {name: reason for name, reason in failed.get(location, {}).items() if name not in names}
	if run.error is not None:
		kept |= {name: str(run.error) for name in load_project(location).pending(names)}
	failed[location] = kept

	if run.error is not None:
		notify("bad", f"Analysis failed: {run.error}")
		return failed, no_update
	if run.cancelled:
		notify(
			"warn",
			"Cancelled. Recordings in flight finished their current step first, and their "
			"finished tables stay on disk.",
		)
		return failed, no_update

	noun = "recording" if len(names) == 1 else "recordings"
	notify("good", f"Analysed {len(names)} {noun} in {run.elapsed:.1f} s")
	return failed, []


# ponytail: local installs only - delete with its _check_update and _update callbacks,
# updater.py and the toast's listener in clientside.js once the app is deployed and users
# no longer update it.
def newer_release() -> str | None:
	"""The latest stable PyPI release if it is newer than the installed one, else None.

	PyPI's ``info.version`` skips pre-releases, so an installed rc above the last stable
	release compares as newer and nothing is offered. None too when PyPI is unreachable.
	"""
	latest = _latest_release()
	return latest if latest and Version(latest) > Version(version("deepecohab")) else None


@cache
def _latest_release() -> str | None:
	try:
		with urllib.request.urlopen("https://pypi.org/pypi/deepecohab/json", timeout=3) as r:
			return json.load(r)["info"]["version"]
	except OSError, ValueError, KeyError:
		# Not cached: a lookup that failed at startup is tried again on the next page load.
		_latest_release.cache_clear()
		return None
