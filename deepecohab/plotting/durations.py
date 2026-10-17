from typing import Literal

import polars as pl

Unit = Literal["seconds", "minutes", "hours", "days"]

_UNITS: frozendict[Unit, tuple[float, str]] = frozendict(
	{
		"days": (86400.0, "d"),
		"hours": (3600.0, "h"),
		"minutes": (60.0, "min"),
		"seconds": (1.0, "s"),
	}
)


def _pick_unit(frame: pl.DataFrame, column: str) -> Unit:
	"""The largest unit in which ``column``'s maximum is still at least 1."""
	largest = frame.select(pl.col(column).dt.total_seconds(fractional=True).max()).item()

	if largest is None:
		return "seconds"

	for unit, (seconds, _) in _UNITS.items():
		if largest >= seconds:
			return unit

	return "seconds"


def _text(column: pl.Expr, unit: Unit) -> pl.Expr:
	"""Format a duration as text, trimmed to the resolution ``unit`` implies."""
	# `dt.round` rejects durations, so trimming goes through a rebuilt duration.
	# Seconds keep milliseconds because chasing_length is sub-second by construction.
	if unit == "seconds":
		trimmed = pl.duration(milliseconds=column.dt.total_milliseconds())
	else:
		trimmed = pl.duration(seconds=column.dt.total_seconds())

	zero = pl.duration(microseconds=0)

	return (
		pl.when(column == zero)
		.then(pl.lit("0s"))
		# Trimming can swallow a small value whole, and "0s" would deny it happened.
		.when(trimmed == zero)
		.then(pl.lit("<1s"))
		.otherwise(trimmed.dt.to_string("polars"))
	)


def to_display(
	frame: pl.DataFrame,
	column: str,
	unit: Unit | Literal["auto"] = "auto",
	label: str = "Time",
) -> tuple[pl.DataFrame, str]:
	"""Replace a duration column with its numeric form and add its hover text.

	Args:
		frame: the collected frame holding ``column``.
		column: name of a ``Duration`` column.
		unit: force a unit, or ``"auto"`` to choose from the data.
		label: axis title stem; the unit is appended.

	Returns:
		The frame with ``column`` as a float in the chosen unit and a ``<column>_text``
		sibling reading ``"3h 7m 4s"``, and the axis title including the unit.
	"""
	chosen: Unit = _pick_unit(frame, column) if unit == "auto" else unit
	seconds, suffix = _UNITS[chosen]
	duration = pl.col(column)
	frame = frame.with_columns(
		(duration.dt.total_seconds(fractional=True) / seconds).alias(column),
		_text(duration, chosen).alias(f"{column}_text"),
	)

	return frame, f"<b>{label} [{suffix}]</b>"
