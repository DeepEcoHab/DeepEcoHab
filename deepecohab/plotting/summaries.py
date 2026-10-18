from typing import Any

import polars as pl

from deepecohab.core.data_model import Layout
from deepecohab.plotting.context import PlotContext


def quality_summary(context: PlotContext) -> dict[str, Any]:
	"""Pooled detection quality: passes ``correct``/``interpolated``/``bad``, ``miss`` in %.

	Returns:
		The pooled counts and miss rate, plus ``antenna_stats``: the same per antenna.
	"""
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


def unknown_time(context: PlotContext) -> pl.DataFrame:
	"""Time each animal spent where antennas could not place it, worst first.

	Returns:
		``animal_id``, ``time_in_position`` and ``share``, a percentage of the recording
		length; every animal of the cohort, at zero when it was never unplaced.
	"""
	# Series.sum() is typed as a broad scalar union; it is a Duration here, same column.
	length = context.table("phase_durations")["duration"].sum().total_seconds()  # ty: ignore[unresolved-attribute]

	return (
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
		.select("animal_id", "time_in_position", "share")
	)


def recording_overview(context: PlotContext) -> dict[str, Any]:
	"""What a recording amounts to, from whichever tables the pipeline has written.

	Returns:
		``days``, ``phases`` and ``mice`` always. ``registrations`` once ``main_df``
		exists; ``visits`` and the ``busiest_cage`` (``position``, ``pct`` of cohort cage
		time, ``cage_type``) once ``activity_df`` does; ``chasings`` and ``top_chaser``
		(``chaser``, ``chasings``; None when there were none) once ``chasings_df`` does.
	"""
	assert context.recording is not None
	overview: dict[str, Any] = {
		"days": context.days_range[1] - context.days_range[0] + 1,
		"phases": context.phase_range[1] - context.phase_range[0] + 1,
		"mice": len(context.animal_ids),
	}

	if "main_df" in context:
		# Lazily: main_df runs to hundreds of thousands of rows, and only their count is read.
		rows = context.recording.load_results("main_df", eager=False).select(pl.len())
		overview["registrations"] = rows.collect().item()

	if "activity_df" in context:
		placed = context.table("activity_df").filter(
			pl.col("position").cast(pl.String) != Layout.UNDEFINED
		)
		overview["visits"] = placed.select(pl.sum("visits_to_position")).item()
		if context.cages:
			busiest = (
				placed.filter(pl.col("position").is_in(context.cages))
				.group_by("position")
				.agg(pl.col("time_in_position").sum().dt.total_seconds().alias("seconds"))
				.with_columns((100 * pl.col("seconds") / pl.col("seconds").sum()).alias("pct"))
				.sort("pct", descending=True)
				.select("position", "pct")
				.row(0, named=True)
			)
			busiest["cage_type"] = next(
				cage.cage_type
				for cage in context.recording.layout.cages
				if cage.name == busiest["position"]
			)
			overview["busiest_cage"] = busiest

	if "chasings_df" in context:
		frame = context.table("chasings_df")
		overview["chasings"] = int(frame["chasings"].sum())
		overview["top_chaser"] = (
			frame.group_by("chaser")
			.agg(pl.sum("chasings"))
			.sort("chasings", descending=True)
			.row(0, named=True)
			if overview["chasings"]
			else None
		)

	return overview
