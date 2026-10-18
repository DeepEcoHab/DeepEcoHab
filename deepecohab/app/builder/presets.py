import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
	from deepecohab import Project

#: The literal a slot substitutes wherever it appears in a preset's state.
EVENT_SLOT = "$event"


@dataclass(frozen=True)
class Preset:
	"""One built-in preset.

	Attributes:
		id: stable key, also used as the saved-preset id prefix.
		name: display name.
		description: what the preset shows and why.
		state: the builder state, with ``EVENT_SLOT`` standing in for the event name
			where the preset asks for one.
		needs_event: the preset asks which event when it opens, and :func:`resolve`
			substitutes the choice for ``EVENT_SLOT``.
		shared_days: restrict the ``day`` filter to the days every recording has.
	"""

	id: str
	name: str
	description: str
	state: dict[str, Any]
	needs_event: bool = False
	shared_days: bool = False


BUILTINS: tuple[Preset, ...] = (
	Preset(
		"time-alone-genotype-sex",
		"Time alone by genotype and sex",
		"One point per animal: its share of observed time spent without company.",
		{
			"measure_as": "rate",
			"kind": "box",
			"channels": {
				"x": ["genotype"],
				"y": ["value"],
				"color": ["sex"],
				"detail": ["recording", "animal_id"],
			},
			"filters": {"metric": {"mode": "pick", "values": ["time_alone"]}},
		},
	),
	Preset(
		"circadian-activity",
		"Circadian activity by genotype",
		"Visits per hour across the 24 hours after phase onset, pooled per genotype.",
		{
			"measure_as": "rate",
			"kind": "line",
			"channels": {"x": ["hour"], "y": ["value"], "color": ["genotype"]},
			"filters": {"metric": {"mode": "pick", "values": ["activity"]}},
		},
	),
	Preset(
		"together-shared-days",
		"Time together over the shared days",
		"Only the days every recording has, worked out from the project when the preset opens.",
		{
			"measure_as": "rate",
			"kind": "line",
			"channels": {
				"x": ["day"],
				"y": ["value"],
				"color": ["genotype"],
				"facet_col": ["sex"],
			},
			"filters": {"metric": {"mode": "pick", "values": ["time_together"]}},
		},
		shared_days=True,
	),
	Preset(
		"light-dark-metrics",
		"Light vs dark, four metrics",
		"Metric on Facet row gives every metric its own axis and unit, so Value never pools units.",
		{
			"measure_as": "rate",
			"kind": "box",
			"channels": {
				"x": ["phase"],
				"y": ["value"],
				"color": ["genotype"],
				"facet_row": ["metric"],
				"detail": ["recording", "animal_id"],
			},
			"filters": {
				"metric": {
					"mode": "pick",
					"values": ["activity", "time_alone", "time_together", "n_chasing"],
				}
			},
		},
	),
	Preset(
		"event-activity",
		"Activity around an event",
		"Asks which event when it opens: hours a bout touches against "
		"the same clock hours on other days.",
		{
			"measure_as": "rate",
			"kind": "box",
			"channels": {
				"x": [EVENT_SLOT],
				"y": ["value"],
				"color": ["genotype"],
				"detail": ["recording", "animal_id"],
			},
			"filters": {"metric": {"mode": "pick", "values": ["activity"]}},
		},
		needs_event=True,
	),
	Preset(
		"chasing-cohort-size",
		"Chasing against cohort size",
		"One point per recording: are larger cohorts more aggressive per available partner?",
		{
			"measure_as": "rate",
			"kind": "scatter",
			"channels": {
				"x": ["n_mice"],
				"y": ["value"],
				"color": ["genotype"],
				"symbol": ["sex"],
				"detail": ["recording"],
			},
			"filters": {"metric": {"mode": "pick", "values": ["n_chasing"]}},
		},
	),
)

BY_ID: frozendict[str, Preset] = frozendict({preset.id: preset for preset in BUILTINS})


def resolve(preset: Preset, project: Project, choice: str | None = None) -> dict[str, Any]:
	"""The preset's state ready to load into the builder.

	Args:
		project: the open project, for its computed filters.
		choice: the event name to substitute for ``EVENT_SLOT``, when the preset asks
			for one.

	Returns:
		A fresh builder state; the preset itself is never mutated.
	"""
	text = json.dumps(preset.state)
	if preset.needs_event and choice:
		text = text.replace(json.dumps(EVENT_SLOT), json.dumps(choice))
	state = json.loads(text)

	if preset.shared_days:
		# Pick-mode values are always strings: ``figure.apply_filters`` matches them
		# against the column cast to ``String``, as the filter widgets do.
		shared = min(recording.timeline.days_range[1] for recording in project.recordings)
		days = [str(day) for day in range(1, shared + 1)]
		state["filters"] = {**state["filters"], "day": {"mode": "pick", "values": days}}

	return state
