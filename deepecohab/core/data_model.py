import copy
import datetime as dt
import graphlib
import json
import logging
import shutil
import threading
import time
import warnings
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from itertools import combinations, pairwise, permutations
from pathlib import Path
from typing import Annotated, Any, ClassVar, Final, Literal, NamedTuple, overload
from zoneinfo import ZoneInfo

import polars as pl
from pydantic import (
	AfterValidator,
	AwareDatetime,
	BaseModel,
	ConfigDict,
	Field,
	PastDate,
	PrivateAttr,
	StringConstraints,
	TypeAdapter,
	ValidationInfo,
	computed_field,
	field_serializer,
	model_validator,
)
lazy from tqdm.auto import tqdm

from deepecohab.core import topology
lazy from deepecohab.core.antenna_analysis import get_prev_ranking
lazy from deepecohab.core.recording_pipeline import event_status

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


def _not_dots(name: str) -> str:
	if not name.strip("."):
		raise ValueError(f"{name!r} is not a usable folder name")
	return name


#: Becomes a folder name, so it must not climb out of its parent or nest.
FolderName = Annotated[
	str,
	StringConstraints(strip_whitespace=True, min_length=1, pattern=r"^[^/\\]+$"),
	AfterValidator(_not_dots),
]


def atomic_write_text(path: Path, text: str) -> None:
	"""Write ``text`` to ``path`` so a crash leaves the old file, never a truncated one."""
	tmp = path.with_name(path.name + ".tmp")
	tmp.write_text(text, encoding="utf-8")
	tmp.replace(path)


CALENDAR_COLUMNS: Final = ("phase", "day", "phase_count", "hour")
"""The columns placing a row on the recording's calendar; every analysis table is keyed by them."""

LEAD_WARNING_THRESHOLD: Final = dt.timedelta(hours=1)
"""How far off a phase onset acquisition may start before adding a recording warns."""


def _elapsed(earlier: dt.datetime, later: dt.datetime) -> dt.timedelta:
	"""Real time between two aware datetimes.

	Both sides go through UTC first: subtracting two datetimes that share a tzinfo
	object gives the wall-clock difference rather than the elapsed one, and ZoneInfo
	instances are cached, so a span crossing a DST transition would come out an hour off.
	"""
	return later.astimezone(dt.UTC) - earlier.astimezone(dt.UTC)


def _real_wall_clock(moment: dt.datetime, zone: ZoneInfo) -> dt.datetime:
	"""*moment* in *zone*, as a wall clock that zone actually has.

	``combine(date, time, tzinfo=zone)`` and ``replace(tzinfo=zone)`` happily mint a
	time a DST jump forward skipped, and ``astimezone`` leaves it be when the zone is
	already that one. Going through UTC pins the instant to the clock that follows it.
	"""
	return moment.astimezone(dt.UTC).astimezone(zone)


class Cage(BaseModel):
	"""EcoHab habitat cage."""

	name: str
	cell_id: str
	cage_type: str
	antennas: list[str]


class Tunnel(BaseModel):
	"""EcoHab habitat tunnel."""

	name: str
	tunnel_no: int
	start_cell_id: str
	end_cell_id: str
	antennas: list[str]


class Layout(BaseModel):
	"""EcoHab habitat layout.

	Defines cages and tunnels within a recording and possible antenna pairs,
	as well as mapping of tunnels (directional to non-directional).
	"""

	UNDEFINED: ClassVar[str] = "undefined"

	cages: list[Cage]
	tunnels: list[Tunnel]
	antenna_combinations: dict[str, str]
	tunnels_map: dict[str, str]

	@model_validator(mode="after")
	def _check_positions(self) -> Layout:
		"""Every position the maps can name has to be a place this layout holds.

		``reindex_onto_grid`` joins on ``positions_non_directional``, so a position that
		is in neither list is not lost loudly - the time spent there silently drops out
		of every table at that join.
		"""
		cages = {cage.name for cage in self.cages}
		tunnels = {tunnel.name for tunnel in self.tunnels}

		if stray := sorted(set(self.antenna_combinations.values()) - cages - set(self.tunnels_map)):
			raise ValueError(
				f"antenna_combinations resolves to {stray}, which is neither a cage of this "
				"layout nor a key of tunnels_map, so time spent there would be dropped."
			)
		if stray := sorted(set(self.tunnels_map.values()) - tunnels):
			raise ValueError(
				f"tunnels_map maps onto {stray}, which are not tunnels of this layout."
			)
		return self

	@model_validator(mode="after")
	def _check_connected(self) -> Layout:
		"""Tunnels join cages, and together they make one habitat.

		A tunnel ending in no cage, or a cage no chain of tunnels leads to, is a layout the
		antennas cannot have been wired to, and the habitat map has nowhere to draw it.
		"""
		cells = {cage.cell_id: cage.name for cage in self.cages}
		ends = {
			cell for tunnel in self.tunnels for cell in (tunnel.start_cell_id, tunnel.end_cell_id)
		}
		if dangling := sorted(ends - set(cells)):
			raise ValueError(f"tunnels end in {dangling}, which are the cell of no cage.")

		neighbours: dict[str, set[str]] = {cell: set() for cell in cells}
		for tunnel in self.tunnels:
			neighbours[tunnel.start_cell_id].add(tunnel.end_cell_id)
			neighbours[tunnel.end_cell_id].add(tunnel.start_cell_id)

		reached: set[str] = set()
		pending = list(cells)[:1]
		while pending:
			cell = pending.pop()
			if cell not in reached:
				reached.add(cell)
				pending.extend(neighbours[cell] - reached)

		if unreached := sorted(cells[cell] for cell in set(cells) - reached):
			raise ValueError(
				f"cages {unreached} are not linked to the rest of the habitat by tunnels."
			)
		return self

	@computed_field
	@property
	def positions_non_directional(self) -> list[str]:
		"""Cage and tunnel names, non-directional, plus the sentinel for unresolved positions."""
		return [c.name for c in self.cages] + [t.name for t in self.tunnels] + [self.UNDEFINED]

	@property
	def cage_names(self) -> list[str]:
		"""Cage names, sorted."""
		return sorted(cage.name for cage in self.cages)

	@property
	def tunnel_names(self) -> list[str]:
		"""Undirected tunnel names, sorted."""
		return sorted(tunnel.name for tunnel in self.tunnels)

	@property
	def tunnel_names_directional(self) -> list[str]:
		"""Directional tunnel positions (e.g. ``c1_c2``), as they appear in ``main_df``.

		These are the keys of ``tunnels_map``: the positions before
		``transforms.remove_tunnel_directionality`` collapses a tunnel's two ends into one.
		"""
		return sorted(self.tunnels_map)


class Animal(BaseModel):
	"""One mouse: the transponder tag its registrations carry, and what results group by.

	``tag`` is the identity the raw data knows; every other field is metadata the
	``animals`` table joins onto any analysis table.
	"""

	tag: NonEmptyStr
	mouse_line: NonEmptyStr
	genotype: NonEmptyStr
	subject_name: NonEmptyStr | None = None
	sex: NonEmptyStr
	date_of_birth: PastDate
	genetic_background: NonEmptyStr
	treatment: NonEmptyStr | None = None
	notes: str


class Cohort(BaseModel):
	"""The animals sharing one recording, and the tag pairings paired measures run over.

	``animal_tags`` is sorted, and that order is what every dense grid, colour
	assignment and pair enumeration downstream is built from.
	"""

	# 63: occupancy and chase winners are tracked as one bit per animal in an Int64 mask.
	animals: list[Animal] = Field(min_length=1, max_length=63)

	@model_validator(mode="after")
	def _check_tags(self) -> Cohort:
		tags = [animal.tag for animal in self.animals]
		if duplicates := sorted({tag for tag in tags if tags.count(tag) > 1}):
			raise ValueError(f"animal tags must be unique, got duplicates of {duplicates}")
		return self

	@computed_field
	@property
	def n_mice(self) -> int:
		"""How many animals the cohort holds."""
		return len(self.animals)

	@property
	def animal_tags(self) -> list[str]:
		"""Contains animal RFID tags sorted alphanumerically."""
		return sorted(animal.tag for animal in self.animals)

	@property
	def animal_combinations(self) -> list[tuple[str, str]]:
		"""Unordered animal pairs (``a < b``), for symmetric measures.

		A meeting is one event whichever animal is named first, so ``pairwise_meetings``
		holds a single cell per pair.
		"""
		return list(combinations(self.animal_tags, 2))

	@property
	def animal_product(self) -> list[tuple[str, str]]:
		"""Ordered animal pairs (``a != b``, both directions), for directed measures.

		Chasings outcomes distinguish the two roles, so A-beats-B and
		B-beats-A are separate cells.
		"""
		return list(permutations(self.animal_tags, 2))


class Timeline(BaseModel):
	"""Timeline of the experiment.

	Contains all information related to the start and end of the experiment, the
	timezone of the computer that recorded the data, the onset of each circadian
	phase, and the ranges of days and phases present in the recording.
	"""

	start_datetime: AwareDatetime
	end_datetime: AwareDatetime
	recording_timezone: ZoneInfo
	phases: dict[Literal["light_phase", "dark_phase"], dt.time]
	start_from: Literal["light_phase", "dark_phase"]
	end_with: Literal["light_phase", "dark_phase"]

	@model_validator(mode="after")
	def _check_timeline(self) -> Timeline:
		if self.end_datetime <= self.start_datetime:
			raise ValueError("end_datetime must be after start_datetime")

		if self.start_from not in self.phases:
			raise ValueError(
				f"start_from is {self.start_from!r} but phases only defines {sorted(self.phases)}."
			)

		if self.end_with not in self.phases or len(self.phases) < 2:
			raise ValueError(
				f"end_with is {self.end_with!r}, which ends where the other phase begins, "
				f"but phases only defines {sorted(self.phases)}."
			)

		if self.experiment_start >= self.experiment_end:
			raise ValueError(
				f"The recording holds no experiment to analyse: it would start at "
				f"{self.experiment_start} ({self.start_from!r} onset) and end at "
				f"{self.experiment_end} (end of {self.end_with!r})."
			)
		return self

	@computed_field
	@property
	def experiment_start(self) -> dt.datetime:
		"""When the experiment proper begins: the ``start_from`` onset nearest acquisition.

		Days and hours are counted from here, so the same day and hour mark the same
		point of the experiment in every recording however acquisition happened to be
		timed. The nearest onset can fall a little before recording began, which leaves
		the first phase short rather than discarding a whole cycle of data.
		"""
		start = self.start_datetime.astimezone(self.recording_timezone)
		onset = self.phases[self.start_from]

		candidates = [
			dt.datetime.combine(
				start.date() + dt.timedelta(days=offset), onset, tzinfo=self.recording_timezone
			)
			for offset in (-1, 0, 1)
		]
		nearest = min(candidates, key=lambda moment: abs(_elapsed(start, moment)))
		return _real_wall_clock(nearest, self.recording_timezone)

	@property
	def discarded_lead(self) -> dt.timedelta:
		"""Data recorded before :attr:`experiment_start`, which the analysis leaves out."""
		return max(_elapsed(self.start_datetime, self.experiment_start), dt.timedelta(0))

	@property
	def unrecorded_lead(self) -> dt.timedelta:
		"""Time between :attr:`experiment_start` and the start of recording, never captured.

		The first phase is short by this much, and no data is dropped.
		"""
		return max(_elapsed(self.experiment_start, self.start_datetime), dt.timedelta(0))

	@computed_field
	@property
	def experiment_end(self) -> dt.datetime:
		"""When the experiment proper ends: the ``end_with`` phase's close nearest acquisition end.

		A phase ends where the other one begins. Like :attr:`experiment_start`, this
		can fall a little after recording stopped, which leaves the last phase short.
		"""
		end = self.end_datetime.astimezone(self.recording_timezone)
		closing = next(phase for phase in self.phases if phase != self.end_with)
		onset = self.phases[closing]

		candidates = [
			dt.datetime.combine(
				end.date() + dt.timedelta(days=offset), onset, tzinfo=self.recording_timezone
			)
			for offset in (-1, 0, 1)
		]
		nearest = min(candidates, key=lambda moment: abs(_elapsed(end, moment)))
		return _real_wall_clock(nearest, self.recording_timezone)

	@property
	def discarded_tail(self) -> dt.timedelta:
		"""Data recorded after :attr:`experiment_end`, which the analysis leaves out."""
		return max(_elapsed(self.experiment_end, self.end_datetime), dt.timedelta(0))

	@property
	def unrecorded_tail(self) -> dt.timedelta:
		"""Time between the end of recording and :attr:`experiment_end`, never captured.

		The last phase is short by this much, and no data is dropped.
		"""
		return max(_elapsed(self.end_datetime, self.experiment_end), dt.timedelta(0))

	@computed_field
	@property
	def days_range(self) -> tuple[int, int]:
		"""Range of experiment days present in the recording."""
		start, end = self.local_span
		return (1, -(-_elapsed(start, end) // dt.timedelta(days=1)))

	@computed_field
	@property
	def phase_range(self) -> tuple[int, int]:
		"""Range of phases present in the recording."""
		return (1, len(self.phase_boundaries()) + 1)

	@property
	def local_span(self) -> tuple[dt.datetime, dt.datetime]:
		"""The analysed window - experiment start to experiment end - in the recording timezone.

		Everything downstream takes its bounds from here, so the lead-in before
		:attr:`experiment_start` and the tail after :attr:`experiment_end` are trimmed
		once, in one place.
		"""
		return self.experiment_start, self.experiment_end

	def phase_boundaries(self) -> list[dt.datetime]:
		"""Phase switch instants strictly inside the recording, in order."""
		start, end = self.local_span
		candidates = (
			dt.datetime.combine(
				start.date() + dt.timedelta(days=n), hhmm, tzinfo=self.recording_timezone
			)
			for n in range((end.date() - start.date()).days + 1)
			for hhmm in set(self.phases.values())
		)
		return sorted(b for b in candidates if start < b < end and self._wall_clock_exists(b))

	@staticmethod
	def _wall_clock_exists(moment: dt.datetime) -> bool:
		"""False for wall-clock times skipped by a DST jump forward."""
		return moment.astimezone(dt.UTC).astimezone(moment.tzinfo) == moment


class Bout(BaseModel):
	"""One stretch of an event, from ``start`` up to but not including ``end``.

	``position`` names the cages or tunnels it was applied in; a bout without one
	applies to the whole habitat.
	"""

	model_config = ConfigDict(extra="forbid")

	start: AwareDatetime
	end: AwareDatetime
	position: list[str] | None = Field(default=None, min_length=1)

	@model_validator(mode="after")
	def _check_bout(self) -> Bout:
		if self.end <= self.start:
			raise ValueError(f"a bout must end after it starts, got {self.start} to {self.end}")
		return self


class Event(BaseModel):
	"""Something the experimenter did during the recording, and every bout of it.

	A stimulus presentation, an injection or a cage swap. Different events may overlap -
	two stimuli presented at once in different cages - but the bouts of one event may
	not, so an entry typed twice fails rather than counting double.
	"""

	model_config = ConfigDict(extra="forbid")

	name: str
	description: str
	bouts: list[Bout] = Field(min_length=1)
	devices: list[str] | None = Field(default=None, min_length=1)

	@model_validator(mode="after")
	def _check_event(self) -> Event:
		ordered = sorted(self.bouts, key=lambda bout: bout.start)
		for earlier, later in pairwise(ordered):
			if later.start < earlier.end:
				raise ValueError(
					f"bouts of {self.name!r} overlap: {earlier.start} to {earlier.end} "
					f"and {later.start} to {later.end}"
				)
		return self


class Device(BaseModel):
	"""An extra device in the habitat, such as a lickometer, read by an antenna or a TTL port.

	Events name the devices they use by ``name``; ``device_type`` groups devices of one
	kind, so two lickometers in different cages can be analysed together.
	"""

	model_config = ConfigDict(extra="forbid")

	name: NonEmptyStr
	device_type: NonEmptyStr
	description: str
	antenna: NonEmptyStr | None = None
	position: str
	TTL_port: NonEmptyStr | None = None

	@model_validator(mode="after")
	def _check_device(self) -> Device:
		if self.antenna is None and self.TTL_port is None:
			raise ValueError(f"device {self.name!r} needs an antenna, a TTL_port or both")
		return self


class Boundary(BaseModel):
	"""A stretch between two recording parts when nothing was recorded, from ``diagnostic.json``.

	``start`` is the last record before it and ``end`` the first after it. ``kind`` is
	``observed_record_gap`` (silence between parts, cause unknown),
	``confirmed_DAQ_inactive`` (a clean stop and restart) or ``abrupt_end``.
	"""

	model_config = ConfigDict(extra="ignore")

	start: AwareDatetime
	end: AwareDatetime
	kind: Literal["observed_record_gap", "confirmed_DAQ_inactive", "abrupt_end"]

	@model_validator(mode="after")
	def _check_boundary(self) -> Boundary:
		if self.end <= self.start:
			raise ValueError(f"a boundary must end after it starts, got {self.start} to {self.end}")
		return self


class AnalysisParams(BaseModel):
	"""Tuning knobs for the analysis steps.

	Attributes:
		minimum_time: minimum continuous co-presence, in seconds, for a meeting to count.
		minimum_time_alone: minimum continuous solitary time, in seconds, for a span to
			count as time alone. Discards the brief gaps left by animals travelling as a
			group and arriving moments apart.
		extrapolation_limit: how long, in seconds, an animal's last known position is
			carried on past its last registration. Silence beyond this is not evidence of
			staying put, so the rest of the window becomes ``Layout.UNDEFINED``.
		chasing_time_window: min and max length, in seconds, of a chasing event.
		use_prev_ranking: start the ranking from the ratings stored with the recording by
			:meth:`Recording.set_prev_ranking`, when it has any, instead of from scratch.
	"""

	minimum_time: float = 2
	minimum_time_alone: float = 10.0
	extrapolation_limit: float = 43200.0
	chasing_time_window: tuple[float, float] = (0.1, 1.2)
	use_prev_ranking: bool = True


class Progress(NamedTuple):
	"""One finished pipeline step, tagged with the recording it came from.

	``step_index`` counts that recording's finished steps, this one included.
	"""

	recording: str
	step: str
	step_index: int


class DataFrameRegistry:
	"""The analysis pipeline: which tables exist, what each needs, and how to build them.

	Steps register at import time via :meth:`register`, so the graph belongs to the
	class. The registry works out the run order and builds each table, so every step
	is a plain ``(recording, params) -> LazyFrame``.
	"""

	_builders: ClassVar[dict[str, Callable[[Recording, AnalysisParams], pl.LazyFrame]]] = {}
	_requires: ClassVar[dict[str, list[str]]] = {}

	@classmethod
	def register(cls, name: str, requires: Sequence[str] = ()) -> Callable:
		"""Register a pipeline step under ``name``.

		Args:
			name: output key, also the ``results/<name>.parquet`` filename.
			requires: data keys this step reads via ``Recording.load_results``; each
				is an edge in the graph :meth:`step_order` sorts.

		Returns:
			The function unchanged, so it stays directly callable.
		"""

		def wrapper(func: Callable) -> Callable:
			cls._builders[name] = func
			cls._requires[name] = list(requires)
			return func

		return wrapper

	@classmethod
	def list_available(cls) -> list[str]:
		"""Every registered data key."""
		return list(cls._builders)

	@classmethod
	def step_order(cls, targets: list[str] | None = None) -> list[str]:
		"""Sort the steps so each one runs after the steps it requires.

		Args:
			targets: run only these steps and their transitive dependencies;
				``None`` covers every registered step.

		Returns:
			Step names in a valid execution order, deterministic between runs.

		Raises:
			KeyError: a target, or a declared requirement, is not registered.
			ValueError: the requirements form a cycle.
		"""
		unknown = {*reqs for reqs in cls._requires.values()} - set(cls._builders)
		if unknown:
			raise KeyError(
				f"Steps declare requirements that no step produces: {sorted(unknown)}. "
				"The module defining them may not have been imported."
			)

		deps: dict[str, set[str]] = {name: set(reqs) for name, reqs in cls._requires.items()}

		if targets is not None:
			if missing := set(targets) - set(deps):
				raise KeyError(f"Unknown analysis step(s): {sorted(missing)}")
			wanted: set[str] = set()
			stack = list(targets)
			while stack:
				step = stack.pop()
				if step in wanted:
					continue
				wanted.add(step)
				stack.extend(deps[step])
			deps = {name: (d & wanted) for name, d in deps.items() if name in wanted}

		# Fed in sorted, so the order is the same between runs.
		sorter = graphlib.TopologicalSorter({name: deps[name] for name in sorted(deps)})
		try:
			return list(sorter.static_order())
		except graphlib.CycleError as cycle:
			raise ValueError(
				f"Cycle detected among analysis steps: {sorted(set(cycle.args[1]))}"
			) from None

	@classmethod
	def build(
		cls, recording: Recording, name: str, params: AnalysisParams, *, overwrite: bool = False
	) -> None:
		"""Build one step's table, unless its parquet is already there and ``overwrite`` is off.

		The steps it requires must have landed first; :meth:`Project.iter_analysis` sees to it.
		"""
		path = recording.results_path / f"{name}.parquet"
		if overwrite or not path.is_file():
			path.parent.mkdir(parents=True, exist_ok=True)
			# Written aside, so a crash never leaves a truncated file that counts as built.
			tmp = path.with_name(path.name + ".tmp")
			cls._builders[name](recording, params).sink_parquet(
				tmp, compression="lz4", engine="streaming"
			)
			tmp.replace(path)


def recording_status(root: Path) -> dict[str, bool]:
	"""Which pipeline steps a recording has already computed, read straight off disk.

	Reads ``results/`` directly rather than going through a ``Recording``, so a
	project whose ``config.json`` fails validation can still report status for that
	recording from the manifest and results folder alone.

	Args:
		root: the recording's own directory, e.g. ``<project_location>/<name>``.

	Returns:
		Every registered step name mapped to whether its parquet exists.
	"""
	results = root / "results"
	return {
		step: (results / f"{step}.parquet").is_file() for step in DataFrameRegistry.step_order()
	}


class Recording(BaseModel):
	"""One recording: raw registrations, plus the metadata that gives them meaning.

	``timeline``, ``cohort`` and ``layout`` are what turn an antenna reading and a
	timestamp into a position, an animal and an experiment hour. Persisted as
	``config.json`` beside ``raw/`` and ``results/``; ``data`` is excluded from that
	snapshot and reattached from ``raw/data.parquet`` on load.
	"""

	model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

	PREV_RANKING: ClassVar[str] = "prev_ranking.parquet"

	name: FolderName
	project_name: NonEmptyStr
	recording_location: NonEmptyStr
	timeline: Timeline
	cohort: Cohort
	layout: Layout
	events: list[Event] = Field(default_factory=list)
	devices: list[Device] = Field(default_factory=list)
	boundaries: list[Boundary] = Field(default_factory=list)
	notes: str
	data: pl.LazyFrame = Field(exclude=True, repr=False)

	_root: Path | None = PrivateAttr(default=None)

	@property
	def data_schema(self) -> pl.Schema:
		"""Required input data schema, in order; columns of `optional_data_schema` may sit among it.

		``inserted`` marks the rows acquisition preprocessing interpolated rather than read.
		"""
		return pl.Schema(
			{
				"datetime": pl.Datetime("us", time_zone=self.timeline.recording_timezone.key),
				"antenna": pl.Categorical(),
				"time_under": pl.Duration("us"),
				"animal_id": pl.Enum(self.cohort.animal_tags),
				"inserted": pl.Boolean(),
			}
		)

	@property
	def optional_data_schema(self) -> pl.Schema:
		"""Input columns a recording may carry, typed when present; no analysis reads them yet."""
		return pl.Schema(
			{
				"internal_board_timestamp": pl.Datetime(
					"us", time_zone=self.timeline.recording_timezone.key
				),
			}
		)

	@property
	def root(self) -> Path:
		"""This recording's directory inside its project.

		Raises:
			ValueError: the recording is not attached to a project, so it has nowhere
				to read or write results.
		"""
		if self._root is None:
			raise ValueError(
				f"Recording {self.name!r} is not attached to a project and so has no "
				"directory. Add it with Project.add_recording(), or open the project "
				"with Project.load()."
			)
		return self._root

	@property
	def results_path(self) -> Path:
		"""Where this recording's analysis tables are written."""
		return self.root / "results"

	@property
	def raw_path(self) -> Path:
		"""The folder holding the data and the files the recording was added from."""
		return self.root / "raw"

	@property
	def data_path(self) -> Path:
		"""This recording's raw antenna registrations."""
		return self.raw_path / "data.parquet"

	@property
	def config_path(self) -> Path:
		"""This recording's config.json, as :meth:`Project.update_notes` et al. write it."""
		return self.root / Project.CONFIG

	def analysed_gaps(self) -> list[Boundary]:
		"""The gaps in recording that fall in the analysed window, clipped to it."""
		begin, stop = self.timeline.local_span
		return [
			copy.replace(gap, start=max(gap.start, begin), end=min(gap.end, stop))
			for gap in self.boundaries
			if gap.end > begin and gap.start < stop
		]

	def computed_tables(self) -> list[str]:
		"""The registered tables this recording has already computed."""
		done = recording_status(self.root)
		return [name for name in DataFrameRegistry.list_available() if done.get(name)]

	@overload
	def load_results(self, key: str, *, eager: Literal[False] = False) -> pl.LazyFrame: ...
	@overload
	def load_results(self, key: str, *, eager: Literal[True]) -> pl.DataFrame: ...
	def load_results(self, key: str, *, eager: bool = False) -> pl.LazyFrame | pl.DataFrame:
		"""Load an analysis table this recording has already produced.

		Args:
			key: registered data key, e.g. ``"main_df"``, or the name of any other table
				written into ``results/`` - an auxiliary analysis sunk there by hand.
			eager: collect the table instead of scanning it lazily.

		Raises:
			KeyError: nothing produces ``key`` - no registered step, no parquet on disk.
			FileNotFoundError: a registered step produces it but has not run yet.
		"""
		path = self.results_path / f"{key}.parquet"

		if not path.is_file():
			available_dfs = DataFrameRegistry.list_available()
			if key not in available_dfs:
				raise KeyError(f"{key!r} is not a registered data key. Available: {available_dfs}")
			raise FileNotFoundError(
				f"{key!r} has not been computed for recording {self.name!r}. "
				"Run Project.run_analysis() first."
			)

		return pl.read_parquet(path) if eager else pl.scan_parquet(path)

	@property
	def prev_ranking(self) -> pl.DataFrame | None:
		"""The ratings this recording's ranking starts from; ``None`` when it starts fresh."""
		path = self.root / self.PREV_RANKING
		return pl.read_parquet(path) if path.is_file() else None

	def set_prev_ranking(self, ranking: pl.DataFrame | pl.LazyFrame | None) -> None:
		"""Store the ratings this recording's ranking starts from, or remove them with ``None``.

		Every later build of the ranking starts from them, unless
		``AnalysisParams.use_prev_ranking`` is off.

		Args:
			ranking: an earlier recording's ranking of the same animals. The full trajectory
				is fine; each animal's last rating is kept.

		Raises:
			ValueError: ``ranking`` lacks a rating column, or holds animals that are not in
				this cohort.
		"""
		path = self.root / self.PREV_RANKING
		if ranking is None:
			path.unlink(missing_ok=True)
			return

		ratings = get_prev_ranking(ranking).collect()
		if foreign := set(ratings["animal_id"]) - set(self.cohort.animal_tags):
			raise ValueError(
				f"{sorted(foreign)} are not in the cohort of {self.name!r}; a previous ranking "
				"must come from a recording of the same animals."
			)
		tmp = path.with_name(path.name + ".tmp")
		ratings.write_parquet(tmp)
		tmp.replace(path)

	@classmethod
	def from_config(cls, config: dict[str, Any], data_path: Path) -> Recording:
		"""Rebuild a recording from its config.json contents, reattaching its parquet."""
		return cls.model_validate(config, context={"data_path": data_path})

	@model_validator(mode="before")
	@classmethod
	def _scan_frame(cls, values: Any, info: ValidationInfo) -> Any:
		ctx = info.context or {}
		if isinstance(values, dict) and "data" not in values and "data_path" in ctx:
			values = {**values, "data": pl.scan_parquet(ctx["data_path"])}
		return values

	@model_validator(mode="after")
	def _check_schema(self, info: ValidationInfo) -> Recording:
		path = Path((info.context or {}).get("data_path", "<data>")).name
		found = self.data.collect_schema()
		required = self.data_schema
		optional = {col: dtype for col, dtype in self.optional_data_schema.items() if col in found}
		expected = {**required, **optional}

		problems = [f"no {col} column" for col in required if col not in found]
		problems += [f"unexpected {col} column" for col in found if col not in expected]
		problems += [
			f"{col} is {found[col]}, expected {dtype}"
			for col, dtype in expected.items()
			if col in found and found[col] != dtype
		]
		if not problems and [col for col in found if col in required] != list(required):
			problems = [f"columns must be in the order {', '.join(required)}"]
		if problems:
			raise ValueError(f"{path}: schema mismatch - {'; '.join(problems)}")
		return self

	@model_validator(mode="after")
	def _check_antennas(self, info: ValidationInfo) -> Recording:
		"""Every antenna the data reads has to be one this layout names.

		``transforms.get_animal_position`` resolves a pair the layout does not name to
		:attr:`Layout.UNDEFINED`, so data and layout disagreeing about how antennas are
		named fails nothing - it quietly yields a recording whose every position is
		undefined. The other direction is left alone: an antenna the layout names but
		that never read is a dead antenna, which ``recording_quality`` reports.
		"""
		path = Path((info.context or {}).get("data_path", "<data>")).name
		named = topology.antennas(self.layout.antenna_combinations)
		read = self.data.select(pl.col("antenna").cast(pl.Utf8).unique()).collect()["antenna"]

		if stray := sorted(set(read.drop_nulls()) - named):
			raise ValueError(
				f"{path}: reads antennas {stray}, which this layout does not name - its "
				f"antennas are {sorted(named)}, so every read there would resolve to "
				f"{Layout.UNDEFINED!r}."
			)
		return self

	@model_validator(mode="after")
	def _check_events(self) -> Recording:
		names = [event.name for event in self.events]
		if duplicates := sorted({name for name in names if names.count(name) > 1}):
			raise ValueError(f"event names must be unique, got duplicates of {duplicates}")

		start, end = self.timeline.local_span
		positions = self.layout.cage_names + self.layout.tunnel_names

		for event in self.events:
			for bout in event.bouts:
				if bout.start < start or bout.end > end:
					raise ValueError(
						f"a bout of {event.name!r}, {bout.start} to {bout.end}, falls outside the "
						f"analysed window {start} to {end}"
					)
				for position in bout.position or ():
					if position not in positions:
						raise ValueError(
							f"a bout of {event.name!r} is in {position!r}, which the layout does "
							f"not have; its positions are {positions}"
						)
		return self

	@model_validator(mode="after")
	def _localize_boundaries(self) -> Recording:
		"""Boundaries in the recording's zone, in order: polars compares datetimes of one zone."""
		zone = self.timeline.recording_timezone
		self.boundaries = sorted(
			(
				copy.replace(
					boundary,
					start=boundary.start.astimezone(zone),
					end=boundary.end.astimezone(zone),
				)
				for boundary in self.boundaries
			),
			key=lambda boundary: boundary.start,
		)
		return self

	@model_validator(mode="after")
	def _check_devices(self) -> Recording:
		names = [device.name for device in self.devices]
		if duplicates := sorted({name for name in names if names.count(name) > 1}):
			raise ValueError(f"device names must be unique, got duplicates of {duplicates}")

		for event in self.events:
			if unknown := sorted(set(event.devices or ()) - set(names)):
				raise ValueError(
					f"event {event.name!r} uses devices {unknown}, which the recording does not "
					f"have; its devices are {names}"
				)

		positions = self.layout.cage_names + self.layout.tunnel_names
		for device in self.devices:
			if device.position not in positions:
				raise ValueError(
					f"device {device.name!r} is in {device.position!r}, which the layout does "
					f"not have; its positions are {positions}"
				)
		return self


#: The files a recording is delivered as, side by side as ``<name>.<suffix>``, mapped to
#: whether it is required. The config is validated into ``config.json`` and the data into
#: ``raw/data.parquet``; every other file, the config as uploaded included, is kept as is in
#: ``raw/``.
RECORDING_FILES: Final = frozendict(
	{
		"config.json": True,
		"data.parquet": True,
		"diagnostic.json": True,
	}
)


class Diagnostic(BaseModel):
	"""What the analysis takes from acquisition preprocessing's ``diagnostic.json``.

	Only the recording boundaries; everything else in the file is left unread.
	"""

	model_config = ConfigDict(extra="ignore")

	recording_boundaries: list[Boundary]


class FailedRecording(NamedTuple):
	"""A recording, or a stray file, that could not be added, with the exception that stopped it."""

	name: str
	error: Exception


def _group_files(
	paths: Iterable[str | Path],
) -> tuple[dict[Path, dict[str, Path]], list[FailedRecording]]:
	"""Sort files named ``<name>.<suffix>`` into one set per recording.

	Suffix and name are matched ignoring case on every OS, so how files pair up does not
	depend on the filesystem they came from. The same file given twice counts once.

	Returns:
		Every complete set, keyed by its resolved folder and the name as first given, with
		its files keyed by suffix; and a failure for every file no suffix in
		`RECORDING_FILES` fits or that has no name before it, every set missing a required
		file, and every set given two different files for one suffix.
	"""
	bases: dict[tuple[Path, str], Path] = {}
	groups: dict[Path, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
	failed = []
	for path in (Path(p).resolve() for p in paths):
		name = path.name.lower()
		suffix = next((s for s in RECORDING_FILES if name.endswith(f".{s}")), None)
		stem = path.name[: -len(suffix) - 1] if suffix else ""
		if suffix is None:
			*others, last = [f"<name>.{s}" for s in RECORDING_FILES]
			reason = f"does not match the file format {', '.join(others)} or {last}"
			failed.append(FailedRecording(path.name, ValueError(reason)))
			continue
		if not stem.strip(". "):
			reason = f"has no recording name before .{suffix}"
			failed.append(FailedRecording(path.name, ValueError(reason)))
			continue
		base = bases.setdefault((path.parent, stem.casefold()), path.parent / stem)
		if path not in (found := groups[base][suffix]):
			found.append(path)

	complete = {}
	for base, files in groups.items():
		if clashes := {s: found for s, found in files.items() if len(found) > 1}:
			reason = "; ".join(
				f"{len(found)} {s.split('.')[0]} files, "
				f"{' and '.join(p.name for p in found)} - keep one"
				for s, found in clashes.items()
			)
			failed.append(FailedRecording(base.name, ValueError(reason)))
		elif missing := [
			s for s, required in RECORDING_FILES.items() if required and s not in files
		]:
			kinds = " or ".join(f"{s.split('.')[0]} file" for s in missing)
			expected = " and ".join(f"{base.name}.{s}" for s in missing)
			present = " and ".join(found[0].name for found in files.values())
			reason = f"no {kinds} - expected {expected} beside {present}"
			failed.append(FailedRecording(base.name, FileNotFoundError(reason)))
		else:
			complete[base] = {s: found[0] for s, found in files.items()}
	return complete, failed


class AddReport(NamedTuple):
	"""Outcome of a batch add.

	``added`` names every recording that went in, ``replaced`` those of them that took the
	place of a same-named one. ``existing`` maps each recording left out because its name is
	taken to its files, ready to be added again with ``overwrite=True``.
	"""

	added: list[str]
	failed: list[FailedRecording]
	existing: dict[str, list[Path]]
	replaced: list[str]


class Project(BaseModel):
	"""The recordings analysed together, and the manifest that keeps them findable.

	The project is persisted as <project_location>/project.json, which maps each
	recording name to its config.json relative to the manifest. Each recording
	owns <project_location>/<recording_name>/, holding config.json, raw/ (the data,
	the config as uploaded and any optional file of `RECORDING_FILES`) and results/. Data is
	reattached from raw/data.parquet on load. Delisted recordings
	keep their folder and are mapped the same way under ``delisted``, unloaded.
	"""

	MANIFEST: ClassVar[str] = "project.json"
	CONFIG: ClassVar[str] = "config.json"
	LOGFILE: ClassVar[str] = "project.log"
	PROJECT_TABLE: ClassVar[str] = "project_table.parquet"

	project_name: str
	experimenter: str
	project_location: Path = Field(exclude=True)
	created_at: dt.datetime
	description: str = ""
	data_catalog: dict[str, Recording] = Field(default_factory=dict)
	delisted: dict[str, str] = Field(default_factory=dict)

	_logger: logging.Logger | None = PrivateAttr(default=None)

	@field_serializer("data_catalog")
	def _catalog_as_pointers(self, catalog: dict[str, Recording]) -> dict[str, str]:
		"""The manifest records where each config.json is, not what's in it."""
		return {name: f"{name}/{self.CONFIG}" for name in catalog}

	@classmethod
	def create(
		cls,
		project_name: str,
		experimenter: str,
		location: Path,
		description: str = "",
	) -> Project:
		"""Creates the project in a project_name folder inside location and writes its manifest.

		Raises:
			ValueError: ``project_name`` cannot be a folder name (a path, or only dots).
		"""
		project_name = TypeAdapter(FolderName).validate_python(project_name)
		location = (Path(location).expanduser() / project_name).resolve()
		location.mkdir(parents=True, exist_ok=True)
		if (location / cls.MANIFEST).exists():
			raise FileExistsError(
				f"A project already exists at {location}; use Project.load() to open it."
			)

		project = cls(
			project_name=project_name,
			experimenter=experimenter,
			project_location=location,
			created_at=dt.datetime.now(dt.UTC),
			description=description,
		)
		project.log.info("created project %r by %s", project_name, experimenter)
		project._save()
		return project

	@classmethod
	def load(cls, location: Path) -> Project:
		"""Reopen a project written by _save(), following the manifest's pointers."""
		location = Path(location).expanduser().resolve()
		raw = cls.read_manifest(location)

		raw["data_catalog"] = {
			name: cls._read_config(location / rel)
			for name, rel in raw.get("data_catalog", {}).items()
		}

		project = cls.model_validate({**raw, "project_location": location})
		# Debug, not info: the app reloads the project after every change it writes, so this
		# would bracket each one with a line for something the user did not do.
		project.log.debug("opened project (%d recordings)", len(project.data_catalog))
		return project

	@classmethod
	def read_manifest(cls, location: Path) -> dict[str, Any]:
		"""The project's ``project.json`` as written, without loading a recording.

		Its keys are this model's fields; ``data_catalog`` and ``delisted`` map each
		recording name to its config.json. It is what a project that fails to load can
		still be listed from.
		"""
		return json.loads((Path(location) / cls.MANIFEST).read_text(encoding="utf-8"))

	@classmethod
	def locate(cls, path: Path) -> Path:
		"""The project folder ``path`` names: the folder itself, or its ``project.json``.

		Raises:
			FileNotFoundError: there is no ``project.json`` there.
		"""
		folder = Path(path).expanduser()
		if folder.name == cls.MANIFEST:
			folder = folder.parent
		if not (folder / cls.MANIFEST).is_file():
			raise FileNotFoundError(
				f"No {cls.MANIFEST} in {folder}. Pick the folder that holds it; it sits next "
				"to one folder per recording."
			)
		return folder.resolve()

	def close(self) -> None:
		"""Release the log file handle."""
		if self._logger is not None:
			for handler in list(self._logger.handlers):
				self._logger.removeHandler(handler)
				handler.close()
			self._logger = None

	def __enter__(self) -> Project:
		return self

	def __exit__(self, *exc_info: object) -> None:
		self.close()

	def __getitem__(self, name: str) -> Recording:
		try:
			return self.data_catalog[name]
		except KeyError:
			raise KeyError(
				f"No recording named {name!r} in project {self.project_name!r}. "
				f"Available: {sorted(self.data_catalog)}"
			) from None

	def __len__(self) -> int:
		return len(self.data_catalog)

	@property
	def recordings(self) -> list[Recording]:
		"""Every recording in the project, in the order they were added."""
		return list(self.data_catalog.values())

	def add_recording(self, *paths: str | Path, overwrite: bool = False) -> Recording:
		"""Adds one recording from its files and updates the manifest.

		Args:
			paths: the recording's files, named ``<name>.<suffix>`` for each suffix in
				`RECORDING_FILES`.
			overwrite: replace a recording of the same name, listed or delisted, and
				discard its results - how a recording is re-added with an edited config.

		Raises:
			ValueError: ``paths`` are not the files of exactly one recording.
			FileExistsError: the name is taken and ``overwrite`` is off.
		"""
		groups, failed = _group_files(paths)
		if failed or len(groups) != 1:
			reasons = [f"{f.name}: {f.error}" for f in failed] or [f"{len(groups)} recordings"]
			raise ValueError(f"Not the files of one recording - {'; '.join(reasons)}")

		(files,) = groups.values()
		recording, replaced = self._add_one(files, overwrite)
		self._save()
		if replaced:
			warnings.warn(
				f"Replaced recording {recording.name!r}; its results were discarded.", stacklevel=2
			)
		return recording

	def add_recordings(self, paths: Iterable[str | Path], *, overwrite: bool = False) -> AddReport:
		"""Adds every recording among ``paths``, grouped by name as in :meth:`add_recording`.

		Recordings are added independently: a failure on one, or a file that belongs to
		none, is logged and warned about, and the rest still go in. The manifest is
		written once, after the batch. With ``overwrite`` off, a recording whose name is
		taken is left out and listed in `AddReport.existing`.
		"""
		groups, failed = _group_files(paths)
		added: list[str] = []
		existing: dict[str, list[Path]] = {}
		replaced: list[str] = []

		for base, files in groups.items():
			try:
				recording, was_replaced = self._add_one(files, overwrite)
			except FileExistsError:
				existing[base.name] = list(files.values())
				continue
			except Exception as exc:
				failed.append(FailedRecording(base.name, exc))
				self.log.exception("failed to add recording from %s", base)
				continue
			added.append(recording.name)
			if was_replaced:
				replaced.append(recording.name)

		if added:
			self._save()
		self.log.info(
			"batch add: %d added (%d replaced), %d already in the project, %d failed",
			len(added),
			len(replaced),
			len(existing),
			len(failed),
		)

		if replaced:
			warnings.warn(
				f"Replaced {len(replaced)} recordings, discarding their results: "
				f"{', '.join(replaced)}",
				stacklevel=2,
			)
		if existing:
			warnings.warn(
				f"{len(existing)} recordings are already in the project and were left as they "
				f"are; pass overwrite=True to replace them: {', '.join(existing)}",
				stacklevel=2,
			)
		if failed:
			details = "\n".join(f"  {f.name}: {type(f.error).__name__}: {f.error}" for f in failed)
			warnings.warn(
				f"{len(failed)} of {len(failed) + len(added)} recordings could not be "
				f"added:\n{details}",
				stacklevel=2,
			)

		return AddReport(added, failed, existing, replaced)

	def update_notes(self, name: str, notes: str, tag: str | None = None) -> None:
		"""Set a recording's notes, or one of its animals', and persist them to its config.json.

		Args:
			name: the recording to annotate.
			notes: the new notes text.
			tag: an animal's tag to set that animal's notes instead of the recording's.

		Raises:
			KeyError: no recording ``name``, or ``tag`` is not one of its animals.
		"""
		recording = self[name]
		if tag is None:
			recording.notes = notes
		else:
			{animal.tag: animal for animal in recording.cohort.animals}[tag].notes = notes
		atomic_write_text(recording.root / self.CONFIG, recording.model_dump_json(indent=2))
		self.log.info("%s: notes updated%s", name, f" for animal {tag}" if tag else "")

	def remove_recording(self, name: str, *, delete_files: bool = False) -> None:
		"""Removes recording from project.

		By default a soft delete - delists from the catalog, leaving config.json
		and the data on disk for :meth:`reinstate_recording`. ``delete_files``
		removes them too.
		"""
		if name not in self.data_catalog:
			raise KeyError(f"No recording named {name!r} in project {self.project_name!r}.")

		if delete_files:
			shutil.rmtree(self.project_location / name)  # raises, leaving the catalog as it was
		else:
			self.delisted[name] = f"{name}/{self.CONFIG}"
		del self.data_catalog[name]

		self._drop_from_project_table(name)
		self.log.warning(
			"removed recording %r (files %s)",
			name,
			"deleted" if delete_files else "kept",
		)
		self._save()

	def _drop_from_project_table(self, name: str) -> None:
		table_path = self.project_location / self.PROJECT_TABLE
		if table_path.is_file():
			remaining = pl.read_parquet(table_path).filter(pl.col("recording") != name)
			if remaining.is_empty():
				table_path.unlink()
			else:
				remaining.write_parquet(table_path, compression="lz4")

	def reinstate_recording(self, name: str) -> Recording:
		"""Brings a delisted recording back from the files it left on disk.

		Its results come back with it; the project table does not until it is generated again.
		"""
		if name not in self.delisted:
			raise KeyError(
				f"No delisted recording named {name!r} in project {self.project_name!r}. "
				f"Delisted: {sorted(self.delisted)}"
			)

		recording = self._read_config(self.project_location / self.delisted[name])
		del self.delisted[name]
		self.data_catalog[name] = recording
		self.log.info("reinstated recording %r", name)
		self._save()
		return recording

	def run_analysis(
		self,
		params: AnalysisParams | None = None,
		names: Iterable[str] | None = None,
		targets: list[str] | None = None,
		*,
		overwrite: bool = False,
		workers: int | None = None,
	) -> None:
		"""Run the analysis pipeline for every recording, showing one progress bar.

		Recordings are analysed concurrently, so the bar counts every step of every
		recording and names the one that landed most recently; with several recordings
		in flight at once there is no single inner loop left to nest a second bar on.
		Steps whose parquet already exists are skipped, so re-running is cheap and
		resumes where an interrupted run left off.

		A recording that fails does not stop the others - the error is raised once the
		rest have finished.

		Args:
			params: tuning knobs for the steps; defaults are used when omitted.
			names: run only these recordings; ``None`` runs all of them.
			targets: run only these steps and their dependencies.
			overwrite: rebuild tables that already exist on disk.
			workers: how many steps to build at once; ``None`` takes half of polars'
				thread pool, see :meth:`iter_analysis`.
		"""
		# Exact, not an estimate: a step that is skipped still reports itself.
		total = len(self._select(names)) * len(DataFrameRegistry.step_order(targets))
		events = self.iter_analysis(params, names, targets, overwrite=overwrite, workers=workers)

		bar = tqdm(events, total=total, desc="analysis", unit="step")
		for progress in bar:
			bar.set_postfix_str(f"{progress.recording}: {progress.step}")

	def generate_project_table(self, names: Iterable[str] | None = None) -> pl.LazyFrame:
		"""Combine every recording's features into one table on a shared timeline.

		Each recording's features are joined to its cohort metadata and tagged with the
		recording they came from, so results can be grouped by genotype, sex, treatment
		or any other recorded attribute across the whole project. Days and hours already
		count from each recording's own experiment start, so the same day and hour mark
		the same point of the experiment in every row.

		``position_type`` names what each row's position is - its cage's ``cage_type``,
		``"tunnel"`` or ``undefined`` - so positions compare across recordings whose
		layouts name them differently. It is null for rows with no position.

		Recordings of different lengths contribute the days they have; nothing is padded
		or truncated to match.

		The table also gains one column per event name declared anywhere in the
		project, plus ``"Any event"``: "During" / "Same hours, other days" / "Other
		hours" for a recording that declares that event, null for one that does not.
		See :func:`recording_pipeline.event_status`.

		Args:
			names: aggregate only these recordings; ``None`` covers the whole project.

		Raises:
			ValueError: the project has no recordings to aggregate.
			FileNotFoundError: a recording has not been analysed yet.

		Returns:
			The table, which is also written to ``project_table.parquet``.
		"""
		recordings = self._select(names)
		if not recordings:
			raise ValueError(f"Project {self.project_name!r} has no recordings to aggregate.")

		event_columns = self.event_names(names)

		frames = []
		for recording in recordings:
			frame = recording.load_results("feature_df").join(
				recording.load_results("animals"), on="animal_id", how="left"
			)

			if event_columns:
				frame = frame.join(
					event_status(recording, event_columns), on=["day", "hour"], how="left"
				)

			layout = recording.layout
			position_types = {cage.name: cage.cage_type for cage in layout.cages}
			position_types |= dict.fromkeys(layout.tunnel_names, "tunnel")

			frames.append(
				frame.with_columns(
					pl.lit(recording.name).alias("recording"),
					pl.lit(recording.cohort.n_mice).cast(pl.UInt16).alias("n_mice"),
					# Both enums are built per recording, so a cohort or a light cycle that
					# differs gives them different categories; strings concatenate cleanly.
					pl.col("animal_id").cast(pl.String),
					pl.col("phase").cast(pl.String),
					pl.col("position").cast(pl.String),
					pl.col("position")
					.cast(pl.String)
					.replace(position_types)
					.alias("position_type"),
				).select("recording", "n_mice", pl.exclude("recording", "n_mice"))
			)

		path = self.project_location / self.PROJECT_TABLE
		tmp = path.with_name(path.name + ".tmp")
		pl.concat(frames, how="vertical").sink_parquet(tmp, compression="lz4", engine="streaming")
		tmp.replace(path)
		self.log.info("wrote project table from %d recordings", len(frames))
		return pl.scan_parquet(path)

	@overload
	def load_project_table(self, *, eager: Literal[False] = False) -> pl.LazyFrame: ...
	@overload
	def load_project_table(self, *, eager: Literal[True]) -> pl.DataFrame: ...
	def load_project_table(self, *, eager: bool = False) -> pl.LazyFrame | pl.DataFrame:
		"""Load the table written by :meth:`generate_project_table`.

		Args:
			eager: collect the table instead of scanning it lazily.

		Raises:
			FileNotFoundError: the table has not been generated yet.
		"""
		path = self.project_location / self.PROJECT_TABLE
		if not path.is_file():
			raise FileNotFoundError(
				f"Project {self.project_name!r} has no project table yet. "
				"Run Project.generate_project_table() first."
			)

		return pl.read_parquet(path) if eager else pl.scan_parquet(path)

	def iter_analysis(
		self,
		params: AnalysisParams | None = None,
		names: Iterable[str] | None = None,
		targets: list[str] | None = None,
		*,
		overwrite: bool = False,
		workers: int | None = None,
		cancel: threading.Event | None = None,
	) -> Iterator[Progress]:
		"""Run the pipeline across recordings as one flat event stream.

		Every step of every recording goes to one pool as soon as the steps it requires
		have landed, so independent steps - of one recording or of several - build side by
		side. Events arrive in the order steps finish, each tagged with its recording and
		with how many of that recording's steps have finished; a step never arrives before
		the steps it requires.

		A recording that raises does not stop the others; the error surfaces here once the
		rest have finished, as it does in :meth:`add_recordings`.

		Args:
			workers: how many steps to build at once; ``None`` takes half of polars'
				thread pool. Polars already spreads every step over all cores, so the gain
				comes from the serial parts between them, while each step in flight adds
				its own peak memory.
			cancel: once set, no further step starts, so the steps in flight still land and
				recordings not yet started never begin.
		"""
		recordings = self._select(names)
		if not recordings:
			return

		params = params or AnalysisParams()
		order = DataFrameRegistry.step_order(targets)
		workers = workers or max(1, pl.thread_pool_size() // 2)
		cancel = cancel or threading.Event()

		# Largest first: the biggest recording sets the run's length, so its steps must not
		# queue behind the small ones for a worker.
		largest_first = sorted(recordings, key=lambda r: r.data_path.stat().st_size, reverse=True)
		queued = [(recording, step) for recording in largest_first for step in order]
		done: dict[str, set[str]] = {recording.name: set() for recording in recordings}
		running: dict[Future, tuple[Recording, str]] = {}
		failures: list[BaseException] = []

		def ready(recording: Recording, step: str) -> bool:
			return set(DataFrameRegistry._requires[step]) <= done[recording.name]

		# Abandoning the generator finishes only the steps in flight.
		with ThreadPoolExecutor(workers) as pool:
			while True:
				if not cancel.is_set():
					for item in [item for item in queued if ready(*item)][: workers - len(running)]:
						queued.remove(item)
						recording, step = item
						future = pool.submit(
							DataFrameRegistry.build, recording, step, params, overwrite=overwrite
						)
						running[future] = item
				if not running:
					break

				finished, _ = wait(running, return_when=FIRST_COMPLETED)
				for future in finished:
					recording, step = running.pop(future)
					if (error := future.exception()) is not None:
						failures.append(error)
						queued = [item for item in queued if item[0] is not recording]
						continue
					done[recording.name].add(step)
					# Logged on this side, so the lazily attached file handler is never raced.
					self.log.info("%s: built %s", recording.name, step)
					yield Progress(recording.name, step, len(done[recording.name]))

		if cancel.is_set():
			self.log.warning("cancelled; steps in flight finished")
		if failures:
			# Callers show one line of this and drop the rest; the log keeps all of it.
			for error in failures:
				self.log.error("analysis failed", exc_info=error)
			raise failures[0]

	def pending(self, names: Iterable[str] | None = None) -> list[str]:
		"""The recordings that miss at least one table of the pipeline.

		Args:
			names: look only at these recordings; ``None`` covers the whole project.
		"""
		return [
			recording.name
			for recording in self._select(names)
			if not all(recording_status(recording.root).values())
		]

	def project_table_stale(self) -> bool:
		"""Whether the project table misses a listed recording or predates any of its results.

		False when the table has not been generated.
		"""
		table = self.project_location / self.PROJECT_TABLE
		if not table.is_file():
			return False
		built = table.stat().st_mtime_ns
		listed = set(
			pl.scan_parquet(table).select(pl.col("recording").unique()).collect().to_series()
		)
		return listed != set(self.data_catalog) or any(
			path.stat().st_mtime_ns > built
			for recording in self.recordings
			for path in recording.results_path.glob("*.parquet")
		)

	def event_counts(self) -> dict[str, int]:
		"""How many recordings declare each event, "Any event" counting recordings with any."""
		counts = dict.fromkeys(self.event_names(), 0)
		for recording in self.recordings:
			declared = {event.name for event in recording.events}
			for name in declared:
				counts[name] += 1
			if declared:
				counts["Any event"] += 1
		return counts

	def event_names(self, names: Iterable[str] | None = None) -> list[str]:
		"""Every event name the recordings declare, plus "Any event"; empty when they declare none.

		Args:
			names: look only at these recordings; ``None`` covers the whole project.
		"""
		declared = {event.name for recording in self._select(names) for event in recording.events}
		return [*sorted(declared), "Any event"] if declared else []

	def _select(self, names: Iterable[str] | None) -> list[Recording]:
		"""The recordings a run covers; the one place that decides which of them run."""
		return [self[name] for name in names] if names is not None else self.recordings

	def _add_one(self, files: dict[str, Path], overwrite: bool) -> tuple[Recording, bool]:
		"""Validate, write, and catalog one recording from its files keyed by suffix.

		Returns the recording and whether it replaced a same-named one. Raises on any
		failure, and with `FileExistsError` when the name is taken and ``overwrite`` is off.
		"""
		metadata = json.loads(files["config.json"].read_text(encoding="utf-8"))
		diagnostic = Diagnostic.model_validate_json(
			files["diagnostic.json"].read_text(encoding="utf-8")
		)
		recording = Recording.from_config(
			{**metadata["recording"], "boundaries": diagnostic.recording_boundaries},
			files["data.parquet"],
		)
		root = self.project_location / recording.name

		if root.exists() and not overwrite:
			raise FileExistsError(
				f"Recording {recording.name!r} is already in project {self.project_name!r}."
			)

		# Replacing discards the results too: re-uploading with an edited config is how a
		# recording's settings change. The old folder is kept aside until the new one lands.
		previous = root.with_name(f"{root.name}.replaced") if root.exists() else None
		if previous:
			shutil.rmtree(previous, ignore_errors=True)
			root.rename(previous)

		target = root / "raw" / "data.parquet"
		try:
			target.parent.mkdir(parents=True)
			(root / "results").mkdir(parents=True)
			recording.data.sink_parquet(target)
			recording.data = pl.scan_parquet(target)  # our copy, not the caller's
			atomic_write_text(root / self.CONFIG, recording.model_dump_json(indent=2))
			for suffix, path in files.items():
				if suffix != "data.parquet":
					path.copy(root / "raw" / suffix)
		except Exception:
			shutil.rmtree(root, ignore_errors=True)  # only ours; the old one is aside
			if previous:
				previous.rename(root)
			raise

		if previous:
			shutil.rmtree(previous, ignore_errors=True)
			self._drop_from_project_table(recording.name)
			self.delisted.pop(recording.name, None)
			self.log.warning("replaced recording %r and discarded its results", recording.name)

		recording._root = root
		self.data_catalog[recording.name] = recording
		self.log.info(
			"added recording %r from %s (%d animals)",
			recording.name,
			files["config.json"],
			recording.cohort.n_mice,
		)

		line = recording.timeline
		if discarded := line.discarded_lead:
			lead = discarded
			offset = f"{discarded} of data recorded before it is left out of the analysis"
		elif unrecorded := line.unrecorded_lead:
			lead = unrecorded
			offset = (
				f"recording started {unrecorded} into it, so that first phase is short by "
				"as much and no data is dropped"
			)
		else:
			lead, offset = dt.timedelta(0), ""

		if offset:
			note = (
				f"the experiment starts at {line.experiment_start}, the nearest "
				f"{line.start_from} onset; {offset}"
			)
			self.log.warning("%r: %s", recording.name, note)
			if lead > LEAD_WARNING_THRESHOLD:
				warnings.warn(f"Recording {recording.name!r}: {note}.", stacklevel=2)

		if discarded := line.discarded_tail:
			trail = discarded
			offset = f"{discarded} of data recorded after it is left out of the analysis"
		elif unrecorded := line.unrecorded_tail:
			trail = unrecorded
			offset = (
				f"recording stopped {unrecorded} before it, so that last phase is short by "
				"as much and no data is dropped"
			)
		else:
			trail, offset = dt.timedelta(0), ""

		if offset:
			note = (
				f"the experiment ends at {line.experiment_end}, the nearest end of the "
				f"{line.end_with}; {offset}"
			)
			self.log.warning("%r: %s", recording.name, note)
			if trail > LEAD_WARNING_THRESHOLD:
				warnings.warn(f"Recording {recording.name!r}: {note}.", stacklevel=2)

		return recording, previous is not None

	@classmethod
	def _read_config(cls, config_path: Path) -> Recording:
		"""Load one recording from its config.json, reattaching the sibling parquet."""
		if not config_path.is_file():
			raise FileNotFoundError(f"{config_path} is listed in the manifest but is missing.")
		if not (config_path.parent / "raw" / "diagnostic.json").is_file():
			raise FileNotFoundError(
				f"Recording {config_path.parent.name!r} has no raw/diagnostic.json, which the "
				"analysis now needs for its recording boundaries and interpolated reads. Add "
				"it again with its <name>.diagnostic.json, passing overwrite=True."
			)

		recording = Recording.from_config(
			json.loads(config_path.read_text(encoding="utf-8")),
			config_path.parent / "raw" / "data.parquet",
		)
		recording._root = config_path.parent
		return recording

	def _save(self) -> None:
		"""Write the pointer manifest to <project_location>/project.json."""
		atomic_write_text(self.project_location / self.MANIFEST, self.model_dump_json(indent=2))
		self.log.info("saved manifest (%d recordings)", len(self.data_catalog))

	@property
	def log(self) -> logging.Logger:
		"""Logger writing to <project_location>/project.log."""
		if self._logger is None:
			path = self.project_location / self.LOGFILE
			logger = logging.getLogger(f"deepecohab.project.{path}")  # unique per project
			if not logger.handlers:
				logger.setLevel(logging.INFO)
				handler = logging.FileHandler(path, encoding="utf-8")
				handler.setFormatter(
					logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")
				)
				logger.addHandler(handler)
			self._logger = logger
		return self._logger


class AnalysisRun:
	"""Analyse recordings of one project on a thread of its own, side by side.

	It starts on construction. The caller runs no thread of its own: it reads
	:meth:`progress` whenever it wants to show it, and :attr:`done` tells it when to stop.
	An exception the analysis raises lands in :attr:`error`.

	Args:
		location: the project folder.
		names: the recordings to build.
	"""

	def __init__(
		self,
		location: Path,
		names: list[str],
		params: AnalysisParams | None = None,
		*,
		overwrite: bool = False,
	) -> None:
		self._steps = len(DataFrameRegistry.step_order())
		self._progress = {name: [0, self._steps] for name in names}
		self._lock = threading.Lock()
		self._cancel = threading.Event()
		self._started = time.perf_counter()
		self.error: Exception | None = None
		self.elapsed = 0.0
		self._thread = threading.Thread(
			target=self._run,
			args=(Path(location), names, params or AnalysisParams(), overwrite),
			daemon=True,
		)
		self._thread.start()

	@property
	def done(self) -> bool:
		"""The analysis has finished, failed or been cancelled."""
		return not self._thread.is_alive()

	@property
	def cancelled(self) -> bool:
		"""Whether :meth:`cancel` was called."""
		return self._cancel.is_set()

	def cancel(self) -> None:
		"""Start no further step; the steps in flight still land."""
		self._cancel.set()

	def progress(self) -> dict[str, list]:
		"""``{recording: [steps done, steps]}``, as of now."""
		with self._lock:
			return dict(self._progress)

	def _run(
		self, location: Path, names: list[str], params: AnalysisParams, overwrite: bool
	) -> None:
		try:
			with Project.load(location) as project:
				for event in project.iter_analysis(
					params, names, overwrite=overwrite, cancel=self._cancel
				):
					with self._lock:
						self._progress[event.recording] = [event.step_index, self._steps]
		except Exception as exc:  # reported by the caller
			self.error = exc
		finally:
			self.elapsed = time.perf_counter() - self._started
