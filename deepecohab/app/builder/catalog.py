import copy
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import polars as pl

Kind = Literal["dimension", "time", "measure"]

#: Columns that carry no grouping or measuring value, so they never become chips.
EXCLUDED: frozenset[str] = frozenset({"notes"})

#: Columns that order or filter rather than measure, regardless of their dtype:
#: counting things (day, phase_count, hour, n_mice) or already-converted duration
#: (age_days) that would otherwise read as a numeric measure.
ORDERED_COLUMNS: frozenset[str] = frozenset({"day", "phase_count", "hour", "n_mice", "age_days"})

AGE = "age_days"
#: What Age can be rounded to, in days per unit; a month is the mean Gregorian one.
AGE_UNITS: frozendict[str, float] = frozendict({"days": 1, "weeks": 7, "months": 30.4375})

#: The long-format triple the rate semantics are built on.
VALUE, EXPOSURE, METRIC = "value", "exposure", "metric"

#: Column -> palette group, for the columns deepecohab always produces. Anything not
#: named here is one of the project's own declared events (see ``prepare``), or falls
#: back to "Recording" - the dtype-based classification below still decides its kind.
GROUPS: frozendict[str, str] = frozendict(
	{
		VALUE: "Measure",
		METRIC: "Measure",
		"day": "Time",
		"phase_count": "Time",
		"hour": "Time",
		"phase": "Time",
		"position": "Position",
		"position_type": "Position",
		"recording": "Recording",
		"n_mice": "Recording",
		"animal_id": "Animal",
		"subject_name": "Animal",
		"sex": "Animal",
		"genotype": "Animal",
		"treatment": "Animal",
		"mouse_line": "Animal",
		"genetic_background": "Animal",
		"age_days": "Animal",
		"date_of_birth": "Animal",
	}
)


@dataclass(frozen=True)
class Field:
	"""One draggable chip: a column of the project table, or a derived measure.

	Attributes:
		name: the column the chip resolves to once the frame is aggregated.
		label: what the chip reads as, and what the axis is titled.
		kind: which shelves will accept it.
		group: which palette section the chip is offered under.
		agg: how it collapses within a group; ``None`` for grouping fields.
		metric: the one metric a single-metric Value chip is read from; ``None`` for
			the pooled Value and every other field.
	"""

	name: str
	label: str
	kind: Kind
	group: str
	agg: Literal["mean", "metric"] | None = None
	metric: str | None = None

	@property
	def discrete(self) -> bool:
		"""Whether dropping this field adds a key to the group-by."""
		return self.agg is None


def metric_names(frame: pl.LazyFrame) -> list[str]:
	"""The metrics the long table holds, sorted; empty when there is no ``metric`` column."""
	if METRIC not in frame.collect_schema():
		return []

	return distinct_values(frame, METRIC)


def prepare(
	frame: pl.LazyFrame, event_names: Sequence[str] = ()
) -> tuple[pl.LazyFrame, list[Field]]:
	"""Make the table plottable and derive the chips the palette offers.

	The table is long: one row per animal-hour *per metric*, so a bare ``value``
	column pools quantities that share no units. ``value`` is one measure and
	``metric`` is the dimension that says which metric it is - filter it to one, or
	facet by it, and the shelves take care of the rest (see ``figure.warnings_for``).
	Each metric is also offered as a chip of its own, read from that metric's rows
	only, so two metrics can share one plot - one on X, the other on Y. ``age``
	becomes whole days, since a birth date is what is recorded but age in days is
	what gets plotted. Every
	remaining column is classified from its dtype rather than by name, so a project
	table that gains a column gains a chip - grouped under "Events" when its name is
	one of ``event_names`` and not a column deepecohab always produces, else under
	whichever ``GROUPS`` names, defaulting to "Recording".

	Args:
		event_names: every event name declared anywhere in the project, plus
			``"Any event"`` - the columns :func:`deepecohab.core.recording_pipeline.
			event_status` added to the table.
	"""
	schema = frame.collect_schema()

	if "age" in schema:
		frame = frame.with_columns(pl.col("age").dt.total_days().alias("age_days")).drop("age")
		schema = frame.collect_schema()

	durations = {
		name
		for name, dtype in schema.items()
		if isinstance(dtype, pl.Duration) and name not in EXCLUDED
	}
	frame = frame.with_columns(
		pl.col(name).dt.total_hours(fractional=True).truediv(24) for name in durations
	)

	fields: list[Field] = []
	if VALUE in schema and EXPOSURE in schema and METRIC in schema:
		fields += [
			Field(VALUE, "Value", "measure", GROUPS[VALUE], "metric"),
			Field(METRIC, "Metric", "dimension", GROUPS[METRIC]),
		]
		# Prefixed, so a metric can never shadow a project column of the same name.
		fields += [
			Field(
				f"{METRIC}:{name}", name.replace("_", " "), "measure", GROUPS[VALUE], "metric", name
			)
			for name in metric_names(frame)
		]

	for name, dtype in schema.items():
		if name in EXCLUDED or name in {VALUE, EXPOSURE, METRIC}:
			continue

		pretty = name.replace("_", " ")
		group = GROUPS.get(name) or ("Events" if name in event_names else "Recording")

		if name in ORDERED_COLUMNS or dtype == pl.Date:
			label = "Age (days)" if name == "age_days" else pretty
			fields.append(Field(name, label, "time", group))
		elif dtype in (pl.String, pl.Boolean) or isinstance(dtype, pl.Enum | pl.Categorical):
			fields.append(Field(name, pretty, "dimension", group))
		elif name in durations:
			fields.append(Field(name, f"mean {pretty} (days)", "measure", group, "mean"))
		elif dtype.is_numeric():
			fields.append(Field(name, f"mean {pretty}", "measure", group, "mean"))

	return frame, fields


def in_age_unit(fields: Sequence[Field], unit: str) -> list[Field]:
	"""``fields`` with Age labelled in the unit it is rounded to."""
	return [
		copy.replace(item, label=f"Age ({unit})") if item.name == AGE else item for item in fields
	]


def distinct_values(frame: pl.LazyFrame, column: str) -> list[str]:
	"""The values a categorical filter can offer for ``column``, sorted, as strings."""
	values = frame.select(pl.col(column).cast(pl.String).unique()).collect().to_series()
	return sorted(value for value in values.to_list() if value is not None)


def value_range(frame: pl.LazyFrame, column: str) -> tuple[float, float]:
	"""The bounds a numeric filter's slider spans: ``column``'s minimum and maximum."""
	bounds = frame.select(
		pl.col(column).min().alias("lo"), pl.col(column).max().alias("hi")
	).collect()
	return float(bounds["lo"][0]), float(bounds["hi"][0])
