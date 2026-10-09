from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Final, Literal, get_args

import polars as pl

from deepecohab.core.data_model import Layout

if TYPE_CHECKING:
	from deepecohab.core.data_model import Recording

Granularity = Literal["day", "phase_count"]

Scope = Literal["all", "cages", "tunnels"]
"""Which positions a plot covers. Cages and tunnels carry the same measures."""

FacetScope = Literal["cages", "tunnels"]
"""``Scope`` without ``"all"``, for the plots that draw one panel per position.

Faceted heatmaps share a single colour axis across their panels, and cage dwell runs to
hours where tunnel dwell runs to seconds - so mixing the two renders the tunnel panels a
uniform dark. Those plots offer one kind at a time instead.
"""

LabelBy = Literal["animal_id", "subject_name"]
"""What an animal is called on an axis, a node or a legend: its tag or its subject name."""

SCOPE_NOUN: Final[dict[str, str]] = {"cages": "cage", "tunnels": "tunnel", "all": "position"}
"""The word a scope goes by in a title or an axis label."""


@dataclass(eq=False)
class PlotContext:
	"""Everything a plot needs that is not a user selection.

	Attributes:
		animal_ids: every cohort tag, sorted; the order colours are assigned in.
		cages: cage names.
		positions: cages and tunnels.
		phases: phase onsets in hours since the ``start_from`` onset, where the hour axis
			begins.
		days_range: first and last experiment day.
		phase_range: first and last phase occurrence.
		tunnels_map: directional tunnel position name to its undirected name, for plots
			reading a raw table that has not been through
			:func:`~deepecohab.core.transforms.remove_tunnel_directionality`.
		recording: the recording whose result parquets the tables are read from. A
			context built by hand instead hands its tables over in ``_loaded``.
		tunnel_lengths_cm: crossing distance in centimetres by undirected tunnel name.
			Speed plots require lengths; missing metadata must not imply a default distance.
			Config integration is pending the upstream tunnel-distance schema.
	"""

	animal_ids: list[str]
	cages: list[str]
	positions: list[str]
	phases: dict[str, float]
	days_range: tuple[int, int]
	phase_range: tuple[int, int]
	tunnels_map: dict[str, str]
	recording: "Recording | None" = None
	_loaded: dict[str, pl.DataFrame] = field(default_factory=dict, repr=False)
	tunnel_lengths_cm: dict[str, float] = field(default_factory=dict)

	@classmethod
	def from_recording(cls, recording: "Recording") -> "PlotContext":
		"""Build a context reading the results of a recording whose pipeline has run."""
		timeline = recording.timeline
		onsets = {
			name: onset.hour + onset.minute / 60 + onset.second / 3600
			for name, onset in timeline.phases.items()
		}

		return cls(
			recording=recording,
			animal_ids=recording.cohort.animal_tags,
			cages=recording.layout.cage_names,
			positions=recording.layout.positions_non_directional,
			phases={
				name: (hours - onsets[timeline.start_from]) % 24 for name, hours in onsets.items()
			},
			days_range=timeline.days_range,
			phase_range=timeline.phase_range,
			tunnels_map=recording.layout.tunnels_map,
		)

	@cached_property
	def animals(self) -> pl.DataFrame:
		"""Cohort metadata, one row per animal."""
		return self.table("animals")

	@property
	def tunnels(self) -> list[str]:
		"""Tunnel names: the positions that are neither cages nor the undefined sentinel.

		Derived rather than stored, since ``positions`` already holds cages, tunnels and
		the sentinel, and a context built by hand would otherwise have to repeat itself.
		"""
		known = {*self.cages, Layout.UNDEFINED}
		return [position for position in self.positions if position not in known]

	def scope_positions(self, scope: Scope) -> list[str]:
		"""The positions a scope selection covers.

		Raises:
			ValueError: ``scope`` is not ``"cages"``, ``"tunnels"`` or ``"all"``.

		Returns:
			Cage and/or tunnel names. The ``undefined`` sentinel is never among them: it
			is not a place, and no measure is attributed to it.
		"""
		match scope:
			case "cages":
				return self.cages
			case "tunnels":
				return self.tunnels
			case "all":
				return [*self.cages, *self.tunnels]
			case _:
				raise ValueError(f"scope must be one of {get_args(Scope)}, got {scope!r}")

	def table(self, key: str) -> pl.DataFrame:
		"""Load and cache the analysis table registered under ``key``."""
		if key not in self._loaded and self.recording is not None:
			self._loaded[key] = self.recording.load_results(key, eager=True)

		return self._loaded[key]

	def __contains__(self, key: str) -> bool:
		"""Whether the step producing ``key`` has run for this recording."""
		return key in self._loaded or (
			self.recording is not None
			and (self.recording.results_path / f"{key}.parquet").is_file()
		)

	def axis_range(self, granularity: Granularity) -> tuple[int, int]:
		"""Full range of the day or phase axis, for an unset selection."""
		return self.phase_range if granularity == "phase_count" else self.days_range
