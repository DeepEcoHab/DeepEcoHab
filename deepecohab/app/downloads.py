import io
import json
import tempfile
import zipfile
from pathlib import Path

import polars as pl
from flask import Blueprint, abort, request, send_file

from deepecohab.app import services
from deepecohab.app.builder import catalog as builder_catalog, figure as builder_figure
from deepecohab.core.data_model import DataFrameRegistry, Project, Recording, recording_status
from deepecohab.plotting import PlotRegistry
from deepecohab.plotting.export import export_figure, styled

bp = Blueprint("downloads", __name__)

_SPOOL = 32 * 1024 * 1024  #: zips this small stay in memory; bigger ones spill to disk
#: Werkzeug refuses form bodies over 500 KB; a long recording's timeline posts megabytes.
_EXPORT_FORM_LIMIT = 512 * 1024 * 1024


@bp.after_request
def _echo_download_token(response):
	"""Hand the page's ``download_token`` back as a cookie, which arrives with the file.

	The hidden download frame gives the page no event when a file starts arriving, so this
	cookie is how the spinner on the button that asked for it knows to stop (clientside.js).
	"""
	# The query string, never the form: re-parsing a body refused as too large would raise
	# again here and turn the 413 into a 500.
	token = request.args.get("download_token")
	if token:
		response.set_cookie("deh-download", token, max_age=60)
	return response


def _project(pid: str) -> Project:
	location = services.resolve_project_path(pid)
	if location is None:
		abort(404, "Unknown project id; open it from Projects first.")
	try:
		return services.load_project(location)
	except Exception as exc:
		abort(404, f"Project failed to load: {exc}")


def _recording(pid: str, name: str) -> Recording:
	project = _project(pid)
	try:
		return project[name]
	except KeyError:
		abort(404, "Unknown recording.")


def _computed_tables(recording: Recording) -> list[str]:
	done = recording_status(recording.root)
	return [name for name in DataFrameRegistry.list_available() if done.get(name)]


@bp.route("/download/project/<pid>/table")
def project_table(pid: str):
	"""The project table, parquet or ``?format=csv`` (converted with :func:`services.csv_ready`)."""
	project = _project(pid)
	path = project.project_location / Project.PROJECT_TABLE
	if not path.is_file():
		abort(404, "Generate the project table first.")
	if request.args.get("format") != "csv":
		return send_file(
			path, as_attachment=True, download_name=f"{project.project_name}__project_table.parquet"
		)
	text = services.csv_ready(pl.read_parquet(path)).write_csv()
	return send_file(
		io.BytesIO(text.encode("utf-8")),
		as_attachment=True,
		download_name=f"{project.project_name}__project_table.csv",
		mimetype="text/csv",
	)


@bp.route("/download/project/<pid>/archive.zip")
def project_archive(pid: str):
	"""Every recording's config and results, and its raw data if ``?raw=1``, as one zip."""
	project = _project(pid)
	include_raw = request.args.get("raw") == "1"
	# Closed by send_file once the response is streamed, not here.
	buffer = tempfile.SpooledTemporaryFile(max_size=_SPOOL)  # noqa: SIM115

	with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
		for recording in project.recordings:
			for path in recording.root.rglob("*"):
				relative = path.relative_to(recording.root)
				if path.is_dir() or (not include_raw and relative.parts[0] == "raw"):
					continue
				zf.write(path, str(Path(recording.name) / relative))

	buffer.seek(0)
	suffix = "_with_raw" if include_raw else ""
	return send_file(
		buffer,
		as_attachment=True,
		download_name=f"{project.project_name}__archive{suffix}.zip",
		mimetype="application/zip",
	)


@bp.route("/download/recording/<pid>/<recording_name>/tables.zip")
def recording_tables(pid: str, recording_name: str):
	"""Every table this recording has computed, as one zip, parquet or ``?format=csv``."""
	recording = _recording(pid, recording_name)
	as_csv = request.args.get("format") == "csv"
	buffer = tempfile.SpooledTemporaryFile(max_size=_SPOOL)  # noqa: SIM115 - closed by send_file

	with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
		for name in _computed_tables(recording):
			path = recording.results_path / f"{name}.parquet"
			if as_csv:
				zf.writestr(f"{name}.csv", services.csv_ready(pl.read_parquet(path)).write_csv())
			else:
				zf.write(path, f"{name}.parquet")

	buffer.seek(0)
	suffix = "_csv" if as_csv else ""
	return send_file(
		buffer,
		as_attachment=True,
		download_name=f"{recording.name}__tables{suffix}.zip",
		mimetype="application/zip",
	)


@bp.route("/download/recording/<pid>/<recording_name>/table/<name>.parquet")
def recording_table(pid: str, recording_name: str, name: str):
	"""One computed analysis table, unconverted."""
	recording = _recording(pid, recording_name)
	if name not in _computed_tables(recording):
		abort(404, "That table has not been computed for this recording.")
	path = recording.results_path / f"{name}.parquet"
	return send_file(path, as_attachment=True, download_name=f"{recording.name}__{name}.parquet")


@bp.route("/download/recording/<pid>/<recording_name>/config.json")
def recording_config(pid: str, recording_name: str):
	"""This recording's config.json, as written by :meth:`Project.update_notes` et al."""
	recording = _recording(pid, recording_name)
	return send_file(
		recording.root / "config.json",
		as_attachment=True,
		download_name=f"{recording.name}__config.json",
	)


@bp.route("/download/recording/<pid>/<recording_name>/raw.parquet")
def recording_raw(pid: str, recording_name: str):
	"""This recording's raw antenna registrations."""
	recording = _recording(pid, recording_name)
	path = recording.root / "raw" / "data.parquet"
	if not path.is_file():
		abort(404, "Raw data not found.")
	return send_file(path, as_attachment=True, download_name=f"{recording.name}__raw.parquet")


@bp.route("/download/recording/<pid>/<recording_name>/cohort.csv")
def recording_cohort(pid: str, recording_name: str):
	"""This recording's cohort, one row per animal."""
	recording = _recording(pid, recording_name)
	frame = pl.DataFrame([animal.model_dump(mode="json") for animal in recording.cohort.animals])
	return send_file(
		io.BytesIO(frame.write_csv().encode("utf-8")),
		as_attachment=True,
		download_name=f"{recording.name}__cohort.csv",
		mimetype="text/csv",
	)


def _safe_filename(name: str) -> str:
	cleaned = "".join(c if c.isalnum() or c in "-_." else "-" for c in name).strip("-")
	return cleaned or "plot"


def _plotted(source: dict) -> dict[str, pl.DataFrame]:
	"""The frames an exported plot was drawn from, rebuilt from the request that drew it.

	Args:
		source: ``{"builder": {location, state}}`` for the plot builder, else a recording
			card's ``{"plot", "request", "x_range"}``.
	"""
	if "builder" in source:
		location, state = source["builder"]["location"], source["builder"]["state"]
		frame, fields = services.builder_frame(location)
		fields = builder_catalog.in_age_unit(fields, state.get("age_unit", "days"))
		data = builder_figure.plotted_data(frame, state, fields)
		return {} if data is None else {"data": data}

	context, values = services.plot_options(source["request"])
	zoom = {"x_range": source["x_range"]} if source.get("x_range") else {}
	return PlotRegistry.data(source["plot"], context, **values, **zoom)


@bp.route("/export", methods=["POST"])
def export_plot():
	"""Fit the posted figure to the requested export size and render it through kaleido."""
	request.max_form_memory_size = _EXPORT_FORM_LIMIT
	try:
		figure = json.loads(request.form["figure"])
		form = json.loads(request.form["params"])
	except (KeyError, ValueError):
		abort(400, "Malformed export request.")

	fig = styled(figure, form.get("style", ""))

	filename = _safe_filename(form.get("filename") or "plot")
	fmt = form.get("format", "svg")

	# Read the rendered image into memory before the temp dir cleans up, rather than
	# handing send_file a path inside it: on Windows the cleanup races the response
	# actually being streamed and fails to delete a file still open for reading.
	with tempfile.TemporaryDirectory() as tmp:
		image_path = Path(tmp) / f"{filename}.{fmt}"
		export_figure(
			fig,
			image_path,
			float(form["width_mm"]),
			float(form["height_mm"]),
			float(form["pt"]),
			fmt,
			int(form.get("dpi", 300)),
			form.get("title") or "" if form.get("title_on") else "",
			bool(form.get("legend", True)),
			bool(form.get("events", True)),
		)
		image_bytes = image_path.read_bytes()

	frames = _plotted(form["data"]) if form.get("csv") and form.get("data") else {}
	tables = {name: frame for name, frame in frames.items() if not frame.is_empty()}
	if not tables:
		return send_file(
			io.BytesIO(image_bytes), as_attachment=True, download_name=f"{filename}.{fmt}"
		)

	buffer = io.BytesIO()
	with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
		zf.writestr(f"{filename}.{fmt}", image_bytes)
		for name, frame in tables.items():
			zf.writestr(
				f"{filename}_{name}.csv" if len(tables) > 1 else f"{filename}.csv",
				services.csv_ready(frame).write_csv(),
			)
	buffer.seek(0)
	return send_file(
		buffer, as_attachment=True, download_name=f"{filename}.zip", mimetype="application/zip"
	)
