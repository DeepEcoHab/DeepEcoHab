"""Shared Hypothesis strategies, frame builders, and reference oracles.

Imported by the test_* modules so every property-based test draws from the same
domains (timezones, phase layouts, animals, positions) and compares against the
same independent Python reference implementations.

tz construction: generated naive datetimes are localized as UTC instants and
converted to the target zone, so DST gaps / ambiguous hours never raise while the
local wall clock still sweeps those transitions.
"""

import datetime as dt
from itertools import pairwise
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
from hypothesis import strategies as st

from deepecohab.core import grids
from deepecohab.core.data_model import (
	Animal,
	Cage,
	Cohort,
	Device,
	Event,
	Layout,
	Recording,
	Timeline,
	Tunnel,
)

# America/New_York covers EDT (and EST); the zones have DST transitions on
# different dates, widening coverage. UTC is the no-DST baseline.
TIMEZONES = ["UTC", "America/New_York", "Europe/Warsaw"]

# Phase onsets, as Timeline.phases carries them. Includes the midnight-boundary
# edge cases that exercise get_phase's wrap-around branch.
PHASE_CONFIGS = [
	{"light_phase": dt.time(7, 0), "dark_phase": dt.time(20, 0)},
	{"light_phase": dt.time(0, 0), "dark_phase": dt.time(13, 0)},
	{"light_phase": dt.time(12, 0), "dark_phase": dt.time(0, 0)},
	{"light_phase": dt.time(22, 0), "dark_phase": dt.time(10, 0)},
]

ANIMALS = ["A", "B", "C", "D", "E", "F"]
CAGES = ["cage_1", "cage_2", "cage_3", "cage_4"]
POSITIONS = [*CAGES, "tunnel_1", "tunnel_2"]
PHASE_NAMES = ["light_phase", "dark_phase"]

#: A recording's ``diagnostic.json`` with no recording boundaries.
MINIMAL_DIAGNOSTIC = '{"recording_boundaries": []}'

_MIN, _MAX = dt.datetime(2023, 1, 1), dt.datetime(2023, 12, 31)

# --- strategies --------------------------------------------------------------
naive_datetimes = st.datetimes(min_value=_MIN, max_value=_MAX)
durations = st.floats(min_value=0, max_value=10 * 60, allow_nan=False, allow_infinity=False)
positive_durations = st.floats(
	min_value=0.1, max_value=10 * 60, allow_nan=False, allow_infinity=False
)
timezones = st.sampled_from(TIMEZONES)
phase_configs = st.sampled_from(PHASE_CONFIGS)
phase_label_lists = st.lists(st.sampled_from(PHASE_NAMES), min_size=1, max_size=40)

# One padding visit: an interval end, its duration, and which animal/position.
visit = st.fixed_dictionaries(
	{
		"end": naive_datetimes,
		"time_spent": durations,
		"animal_id": st.sampled_from(ANIMALS),
		"position": st.sampled_from(POSITIONS),
	}
)

# One presence event for time-alone: an animal in a cage from start for duration.
event = st.fixed_dictionaries(
	{
		"animal_id": st.sampled_from(ANIMALS),
		"position": st.sampled_from(CAGES),
		"start": naive_datetimes,
		"duration": positive_durations,
	}
)


# --- builders ----------------------------------------------------------------
def aware(naive: list[dt.datetime], tz: str) -> pl.Series:
	"""tz-aware Datetime series built via UTC so DST never raises."""
	return (
		pl.Series("datetime", naive)
		.dt.replace_time_zone("UTC")
		.dt.convert_time_zone(tz)
		.cast(pl.Datetime("us", tz))
	)


def seconds(values: list[float]) -> pl.Series:
	"""Durations from float seconds, as main_df and padded_df carry time_spent."""
	return pl.Series(
		"time_spent", [dt.timedelta(seconds=v) for v in values], dtype=pl.Duration("us")
	)


def padding_frame(rows: list[dict], tz: str) -> pl.LazyFrame:
	"""Frame shaped for transforms.split_visits_on_cells, one row per visit."""
	return pl.LazyFrame(
		{
			"animal_id": pl.Series([r["animal_id"] for r in rows], dtype=pl.Enum(ANIMALS)),
			"position": pl.Series([r["position"] for r in rows], dtype=pl.Categorical),
			"datetime": aware([r["end"] for r in rows], tz),
			"time_spent": seconds([r["time_spent"] for r in rows]),
			"time_under": pl.Series(
				[dt.timedelta(seconds=r["time_spent"]) for r in rows], dtype=pl.Duration("us")
			),
		}
	)


def time_alone_frame(events: list[dict], tz: str) -> pl.DataFrame:
	"""Eager frame shaped for _get_time_alone (datetime is the interval END)."""
	ends = [e["start"] + dt.timedelta(seconds=e["duration"]) for e in events]
	return pl.DataFrame(
		{
			"animal_id": pl.Series([e["animal_id"] for e in events], dtype=pl.Enum(ANIMALS)),
			"position": pl.Series([e["position"] for e in events], dtype=pl.Categorical),
			"datetime": aware(ends, tz),
			"time_spent": seconds([float(e["duration"]) for e in events]),
		}
	)


# --- reference oracles -------------------------------------------------------
def expected_time_of_day(moment: dt.datetime, origin: dt.datetime) -> dt.time:
	"""The wall clock get_phase reads, with DST measured against the experiment origin.

	The recording is treated as running on the offset it started with, so an instant in
	a different DST regime is shifted back before its time of day is taken.
	"""
	shift = (moment.dst() or dt.timedelta(0)) - (origin.dst() or dt.timedelta(0))
	return (moment.astimezone(dt.UTC) - shift).astimezone(moment.tzinfo).time()


def expected_phase(local_time: dt.time, phases: dict[str, dt.time]) -> str:
	"""Independent reimplementation of get_phase's mapping for one wall-clock time.

	The latest boundary at or before the time wins; before the first boundary it
	wraps to the last phase. Mirrors the when/then chain in grids.get_phase.
	"""
	boundaries = sorted((start, name) for name, start in phases.items())
	result = boundaries[-1][1]
	for start, name in boundaries:
		if local_time >= start:
			result = name
	return result


def expected_phase_count(labels: list[str]) -> list[int]:
	"""Reference for phase numbering: the global ordinal of each contiguous run,
	regardless of phase type (1,1,2,3,4... for an alternating L/D sequence).
	"""
	out: list[int] = []
	count = 0
	prev: object = None
	for label in labels:
		if label != prev:
			count += 1
		out.append(count)
		prev = label
	return out


def elapsed_hours(moments: list[dt.datetime], origin: dt.datetime) -> list[int]:
	"""Whole hours from the experiment origin to each instant, as the grid counts them.

	Both sides go through UTC first: subtracting two aware datetimes that share a tzinfo
	object gives the wall-clock difference, not the elapsed one, and ZoneInfo instances
	are cached - so a span crossing a DST transition would come out an hour long.
	"""
	origin_utc = origin.astimezone(dt.UTC)
	return [(moment.astimezone(dt.UTC) - origin_utc) // dt.timedelta(hours=1) for moment in moments]


def expected_day(moments: list[dt.datetime], origin: dt.datetime) -> list[int]:
	"""Reference for get_day: 1-indexed 24h block of elapsed time from the origin."""
	return [hours // 24 + 1 for hours in elapsed_hours(moments, origin)]


# === additional infra for the utils-coverage expansion =======================

DIRECTIONAL_TUNNELS = ["c1_c2", "c2_c1", "c2_c3", "c3_c2"]
TUNNEL_POSITIONS = ["tunnel_1", "tunnel_2", "tunnel_3", "tunnel_4"]

# --- additional strategies ---------------------------------------------------
# datetimes truncated to whole seconds, so consecutive diffs are integral.
whole_second_datetimes = naive_datetimes.map(lambda d: d.replace(microsecond=0))

# antenna_combinations: antenna-pair key -> position (cages + directional tunnels)
antenna_combinations = st.dictionaries(
	keys=st.from_regex(r"[0-9]_[0-9]", fullmatch=True),
	values=st.sampled_from(CAGES + DIRECTIONAL_TUNNELS),
	min_size=1,
	max_size=8,
)

# tunnels map: directional tunnel -> undirected position. A layout always has at
# least one tunnel, so an empty map is not a recording worth generating.
tunnels_maps = st.dictionaries(
	keys=st.sampled_from(DIRECTIONAL_TUNNELS),
	values=st.sampled_from(TUNNEL_POSITIONS),
	min_size=1,
	max_size=4,
)

# A cohort always has at least one animal, so an empty draw is not a valid recording.
animal_id_lists = st.lists(st.sampled_from(ANIMALS), min_size=1, max_size=6, unique=True)


# --- additional builders -----------------------------------------------------
def grouped_event_frame(rows: list[dict], tz: str) -> pl.LazyFrame:
	"""Frame with animal_id + datetime in the given row order (NOT sorted), for
	calculate_time_spent tests. rows: {"animal_id", "end"}.
	"""
	return pl.LazyFrame(
		{
			"animal_id": pl.Series([r["animal_id"] for r in rows], dtype=pl.Enum(ANIMALS)),
			"datetime": aware([r["end"] for r in rows], tz),
		}
	)


# --- additional reference oracles --------------------------------------------
def expected_hour(moments: list[dt.datetime], origin: dt.datetime) -> list[int]:
	"""Reference for get_hour: hour within the elapsed 24h block from the origin."""
	return [hours % 24 for hours in elapsed_hours(moments, origin)]


def expected_time_spent(animals: list[str], datetimes: list[dt.datetime]) -> list[dt.timedelta]:
	"""Reference for calculate_time_spent: per-animal consecutive gap in row order;
	the first occurrence of each animal is zero.
	"""
	last: dict[str, dt.datetime] = {}
	out: list[dt.timedelta] = []
	for animal, t in zip(animals, datetimes, strict=False):
		out.append(dt.timedelta(0) if animal not in last else t - last[animal])
		last[animal] = t
	return out


# === antenna_analysis fixtures ===============================================
# Shared recording + frame builders for the analysis steps in
# deepecohab.core.antenna_analysis. The steps read their input table via
# Recording.load_results, so tests monkeypatch that to return one of these
# hand-built frames and call the step directly.

# A small linear EcoHAB layout: four cages in a row joined by three tunnels.
ANALYSIS_CAGES = ["cage_1", "cage_2", "cage_3", "cage_4"]
# Directional tunnel positions as they appear in main_df (before
# remove_tunnel_directionality collapses the two ends).
ANALYSIS_DIRECTIONAL = ["c1_c2", "c2_c1", "c2_c3", "c3_c2", "c3_c4", "c4_c3"]
# Directional tunnel -> undirected tunnel (Layout.tunnels_map).
ANALYSIS_TUNNELS_MAP = {
	"c1_c2": "tunnel_1",
	"c2_c1": "tunnel_1",
	"c2_c3": "tunnel_2",
	"c3_c2": "tunnel_2",
	"c3_c4": "tunnel_3",
	"c4_c3": "tunnel_3",
}
ANALYSIS_UNDIRECTED_TUNNELS = ["tunnel_1", "tunnel_2", "tunnel_3"]
ANALYSIS_POSITIONS = sorted([*ANALYSIS_CAGES, *ANALYSIS_UNDIRECTED_TUNNELS, "undefined"])
# antenna_combinations: Layout.tunnel_names_directional comes from tunnels_map, so
# the directional tunnels must appear here too for the two to agree.
ANALYSIS_ANTENNA_COMBINATIONS = {
	"1_1": "cage_1",
	"2_2": "cage_2",
	"3_3": "cage_3",
	"4_4": "cage_4",
	"1_2": "c1_c2",
	"2_1": "c2_c1",
	"2_3": "c2_c3",
	"3_2": "c3_c2",
	"3_4": "c3_c4",
	"4_3": "c4_c3",
}

# Onsets chosen so that a midnight start_datetime is itself the light_phase onset:
# the experiment origin then coincides with the recording start, which keeps the
# fixtures' day and hour equal to calendar day and wall-clock hour.
DEFAULT_PHASES = {"light_phase": dt.time(0, 0), "dark_phase": dt.time(12, 0)}


def make_layout(antenna_combinations: dict[str, str], tunnels_map: dict[str, str]) -> Layout:
	"""A Layout carrying the given geometry, with placeholder cage/tunnel metadata.

	Cage and tunnel names are derived from the maps, so a test states its geometry
	once as antenna pairs and gets a consistent layout back. A pair resolving to a
	directional tunnel the given ``tunnels_map`` does not name is dropped rather than
	carried: the Layout validator rejects such a position, and a test drawing a partial
	tunnels_map is asking for the smaller habitat, not a broken one.
	"""
	antenna_combinations = {
		pair: position
		for pair, position in antenna_combinations.items()
		if "cage" in position or position in tunnels_map
	}
	cage_names = sorted({pos for pos in antenna_combinations.values() if "cage" in pos})
	tunnel_names = sorted(set(tunnels_map.values()))

	return Layout(
		cages=[
			Cage(name=name, cell_id=f"C{index}", cage_type="standard", antennas=[])
			for index, name in enumerate(cage_names, start=1)
		],
		tunnels=[
			Tunnel(
				name=name,
				tunnel_no=index,
				start_cell_id=f"C{index}",
				end_cell_id=f"C{index + 1}",
				antennas=[],
			)
			for index, name in enumerate(tunnel_names, start=1)
		],
		antenna_combinations=antenna_combinations,
		tunnels_map=tunnels_map,
	)


def make_cohort(animal_ids: list[str]) -> Cohort:
	"""A Cohort of the given tags, with placeholder animal metadata."""
	return Cohort(
		animals=[
			Animal(
				tag=tag,
				mouse_line="test_line",
				genotype="WT",
				subject_name=f"ID{index:02d}",
				sex="Female",
				date_of_birth=dt.date(2023, 1, 1),
				genetic_background="C57B",
				treatment="no",
				notes="",
			)
			for index, tag in enumerate(animal_ids, start=1)
		],
	)


def analysis_recording(
	animal_ids: list[str] | None = None,
	tz: str = "UTC",
	start: str = "2023-05-24 00:00:00",
	finish: str = "2023-05-26 23:00:00",
	phases: dict[str, dt.time] | None = None,
	start_from: str = "light_phase",
	end_with: str = "dark_phase",
	antenna_combinations: dict[str, str] | None = None,
	tunnels_map: dict[str, str] | None = None,
	root: Path | None = None,
	events: list[Event] | None = None,
	devices: list[Device] | None = None,
) -> Recording:
	"""A fully-populated Recording for the antenna_analysis steps.

	Carries everything the grid helpers need (timeline, cohort) plus the linear
	layout above, and any ``events`` and ``devices`` declared on it. ``root`` is where results would
	be written; tests that monkeypatch ``load_results`` can leave it unset.
	"""
	animal_ids = animal_ids or ["A", "B", "C"]
	zone = ZoneInfo(tz)

	recording = Recording(
		name="test_recording",
		project_name="test_project",
		recording_location="test",
		timeline=Timeline(
			start_datetime=dt.datetime.fromisoformat(start).replace(tzinfo=zone),
			end_datetime=dt.datetime.fromisoformat(finish).replace(tzinfo=zone),
			recording_timezone=zone,
			phases=phases or DEFAULT_PHASES,
			start_from=start_from,
			end_with=end_with,
		),
		cohort=make_cohort(animal_ids),
		layout=make_layout(
			antenna_combinations or ANALYSIS_ANTENNA_COMBINATIONS,
			tunnels_map or ANALYSIS_TUNNELS_MAP,
		),
		events=events or [],
		devices=devices or [],
		notes="",
		data=pl.LazyFrame(
			schema={
				"datetime": pl.Datetime("us", time_zone=tz),
				"antenna": pl.Categorical(),
				"time_under": pl.Duration("us"),
				"animal_id": pl.Enum(sorted(animal_ids)),
				"inserted": pl.Boolean(),
			}
		),
	)
	if root is not None:
		recording._root = Path(root)
	return recording


# The shipped EcoHAB layouts are rings rather than the linear fixture above: N cages
# joined head to tail by N tunnels, with an antenna at each tunnel mouth. Antennas are
# numbered 1..2N around the ring starting at the first tunnel, so tunnel k has its
# mouths at antennas 2k-1 and 2k and cage k sits between antennas 2k-2 (2N for cage 1)
# and 2k-1. Both shipped sizes are reproduced exactly by ring_layout below: the default
# 4 cages / 8 antennas and the field 8 cages / 16 antennas.
RING_LAYOUTS = {
	"default": (["1", "2", "3", "4"], "tunnel_{n}"),
	"field": (["A", "B", "C", "D", "E", "F", "G", "H"], "tunnel{n}"),
}


def ring_layout(cages: list[str], tunnel_name: str = "tunnel_{n}") -> tuple[dict, dict]:
	"""Build ``antenna_combinations`` and ``tunnels_map`` for a ring of cages.

	A read pair spanning a tunnel mouth maps to the directional tunnel (``cA_cB``); the
	four pairs among a cage's own two antennas map to that cage. This is the geometry a
	repeat read means something in: both of a cage's antennas sit at a tunnel mouth, so
	reading one twice is the animal poking into that tunnel and backing out.

	Args:
		cages: cage labels in ring order, as they appear in ``cage_<label>``.
		tunnel_name: format for the undirected tunnel name, given ``n`` (1-indexed).

	Returns:
		(antenna_combinations, tunnels_map) as the layout carries them.
	"""
	count = len(cages)
	combinations: dict[str, str] = {}
	tunnels: dict[str, str] = {}

	for index, cage in enumerate(cages):
		following = cages[(index + 1) % count]
		# Tunnel index+1, entered at antenna 2*index+1 from `cage` and 2*index+2 from `following`.
		near, far = 2 * index + 1, 2 * index + 2
		combinations[f"{near}_{far}"] = f"c{cage}_c{following}"
		combinations[f"{far}_{near}"] = f"c{following}_c{cage}"
		tunnels[f"c{cage}_c{following}"] = tunnel_name.format(n=index + 1)
		tunnels[f"c{following}_c{cage}"] = tunnel_name.format(n=index + 1)

		# Cage `cage` is held between the far antenna of the previous tunnel and `near`.
		entry = 2 * index if index else 2 * count
		for first, second in ((entry, near), (near, entry), (entry, entry), (near, near)):
			combinations[f"{first}_{second}"] = f"cage_{cage}"

	return combinations, tunnels


def ring_recording(
	animal_ids: list[str] | None = None,
	section: str = "default",
	**overrides,
) -> Recording:
	"""An analysis recording carrying one of the ring layouts.

	``ANALYSIS_ANTENNA_COMBINATIONS`` above is a linear layout with one antenna per cage,
	where a repeat read means nothing in particular. The shipped layouts are rings with
	the antennas at the tunnel mouths, so a repeat read there is the animal poking into a
	tunnel and backing out -- the only trace a tube-test retreat leaves, and what
	``auxiliary_analysis.tube_test._resolve_repeat_reads`` decodes. Tests that exercise
	that path need this geometry rather than the linear one.

	Args:
		animal_ids: cohort, as for :func:`analysis_recording`.
		section: which ring to use, ``"default"`` (4 cages, 8 antennas) or ``"field"``
			(8 cages, 16).
		**overrides: passed through to :func:`analysis_recording` (tz, start, finish,
			phases, root).

	Returns:
		A recording whose geometry is the ring and whose window, timezone and cohort come
		from :func:`analysis_recording`.
	"""
	combinations, tunnels = ring_layout(*RING_LAYOUTS[section])

	return analysis_recording(
		animal_ids=animal_ids, antenna_combinations=combinations, tunnels_map=tunnels, **overrides
	)


def antenna_reads(
	animal: str, reads: list[tuple[int, dt.datetime]], recording: Recording
) -> list[dict]:
	"""Turn a raw antenna sequence into main_df rows the way the pipeline would.

	``reads`` is ``[(antenna, datetime)]`` in chronological order for one animal.
	``position`` is looked up from the previous and current antenna exactly as
	``transforms.get_animal_position`` does (``undefined`` for the unpaired first read
	and for any unmapped pair) and ``time_spent`` is the gap to the previous read. This
	lets a test state "poked into the tunnel and backed out" as the repeat read it
	actually is, instead of hand-writing a position the raw data could never carry.

	Args:
		animal: animal_id the reads belong to.
		reads: chronological ``(antenna, datetime)`` pairs.
		recording: recording carrying the antenna map to resolve positions with.

	Returns:
		Row dicts ready for :func:`main_df_frame`.
	"""
	antenna_combinations = recording.layout.antenna_combinations
	rows: list[dict] = []
	previous_antenna, previous_time = "missing", None

	for antenna, timestamp in reads:
		rows.append(
			{
				"animal_id": animal,
				"antenna": antenna,
				"position": antenna_combinations.get(f"{previous_antenna}_{antenna}", "undefined"),
				"datetime": timestamp,
				"time_spent": 0.0
				if previous_time is None
				else (timestamp - previous_time).total_seconds(),
			}
		)
		previous_antenna, previous_time = antenna, timestamp

	return rows


def at(*args: int, tz: str = "UTC") -> dt.datetime:
	"""Construct a zone-aware datetime in the given timezone."""
	return dt.datetime(*args, tzinfo=ZoneInfo(tz))


def main_df_frame(rows: list[dict], recording: Recording) -> pl.LazyFrame:
	"""Build a main_df-shaped LazyFrame from plain row dicts.

	Each row needs ``animal_id``, ``position`` (directional), ``datetime``
	(tz-aware) and ``time_spent`` in seconds; ``antenna`` defaults to 0. The
	calendar columns are derived exactly as the real pipeline derives them, so the
	frame matches what the analysis steps consume.
	"""
	frame = pl.DataFrame(
		{
			"animal_id": pl.Series(
				[r["animal_id"] for r in rows], dtype=pl.Enum(recording.cohort.animal_tags)
			),
			"antenna": pl.Series([str(r.get("antenna", 0)) for r in rows], dtype=pl.Categorical),
			"position": pl.Series([r["position"] for r in rows], dtype=pl.Categorical),
			"datetime": pl.Series("datetime", [r["datetime"] for r in rows]),
			"time_spent": seconds([float(r["time_spent"]) for r in rows]),
		}
	)
	return (
		frame.lazy()
		.sort("datetime")
		.with_columns(
			grids.get_phase(recording), grids.get_day(recording), grids.get_hour(recording)
		)
		.pipe(grids.assign_phase_count, recording)
	)


def match_df_frame(rows: list[dict], recording: Recording) -> pl.LazyFrame:
	"""Build a match_df-shaped LazyFrame (the calculate_matches output) from rows.

	Each row needs ``winner``, ``loser``, ``position`` (directional tunnel) and
	``datetime`` (tz-aware). The calendar columns are derived from the datetime, with
	``phase_count`` taken from the time grid so it aligns with the reindex grid used
	downstream. With no rows, an empty frame carrying the full typed schema is
	returned.
	"""
	animals = pl.Enum(recording.cohort.animal_tags)
	timezone = recording.timeline.recording_timezone.key

	if not rows:
		return pl.LazyFrame(
			schema={
				"winner": animals,
				"loser": animals,
				"position": pl.Categorical,
				"datetime": pl.Datetime("us", timezone),
				"chasing_length": pl.Duration("us"),
				"phase": pl.Enum(list(recording.timeline.phases)),
				"day": pl.UInt16,
				"hour": pl.UInt8,
				"phase_count": pl.UInt16,
			}
		)

	frame = pl.DataFrame(
		{
			"winner": pl.Series([r["winner"] for r in rows], dtype=animals),
			"loser": pl.Series([r["loser"] for r in rows], dtype=animals),
			"position": pl.Series([r["position"] for r in rows], dtype=pl.Categorical),
			"datetime": pl.Series("datetime", [r["datetime"] for r in rows]),
			"chasing_length": seconds([float(r.get("chasing_length", 0.5)) for r in rows]),
		}
	)
	return (
		frame.lazy()
		.with_columns(
			grids.get_phase(recording), grids.get_day(recording), grids.get_hour(recording)
		)
		.pipe(grids.assign_phase_count, recording)
	)


def padded_df_frame(rows: list[dict], recording: Recording) -> pl.LazyFrame:
	"""Build a padded_df-shaped LazyFrame from plain row dicts.

	Each row needs ``animal_id``, ``position`` (undirected or cage), ``datetime``
	(the interval END, tz-aware) and ``time_spent`` in seconds; ``interpolated``
	defaults to False. The calendar columns are derived from the piece's START, as
	``split_on_cell_boundaries`` derives them from the piece start - a piece belongs
	to the bin it began in, so a fixture that crosses a boundary has to agree.

	With no rows, an empty frame carrying the full typed schema is returned, as
	:func:`match_df_frame` does.
	"""
	if not rows:
		return pl.LazyFrame(
			schema={
				"animal_id": pl.Enum(recording.cohort.animal_tags),
				"position": pl.Categorical,
				"datetime": pl.Datetime("us", recording.timeline.recording_timezone.key),
				"time_spent": pl.Duration("us"),
				"interpolated": pl.Boolean,
				"phase": pl.Enum(list(recording.timeline.phases)),
				"day": pl.UInt16,
				"hour": pl.UInt8,
				"phase_count": pl.UInt16,
			}
		)

	frame = pl.DataFrame(
		{
			"animal_id": pl.Series(
				[r["animal_id"] for r in rows], dtype=pl.Enum(recording.cohort.animal_tags)
			),
			"position": pl.Series([r["position"] for r in rows], dtype=pl.Categorical),
			"datetime": pl.Series("datetime", [r["datetime"] for r in rows]),
			"time_spent": seconds([float(r["time_spent"]) for r in rows]),
			"interpolated": pl.Series(
				[bool(r.get("interpolated", False)) for r in rows], dtype=pl.Boolean
			),
		}
	)
	return (
		frame.lazy()
		.sort("datetime")
		.with_columns((pl.col("datetime") - pl.col("time_spent")).alias("__start"))
		.with_columns(
			grids.get_phase(recording, "__start"),
			grids.get_day(recording, "__start"),
			grids.get_hour(recording, "__start"),
		)
		.pipe(grids.assign_phase_count, recording)
		.drop("__start")
	)


def expected_matches(
	rows: list[dict], recording: Recording, window: tuple[float, float]
) -> list[tuple]:
	"""Brute-force reference for ``antenna_analysis.calculate_matches``.

	Quadratic and literal: every tunnel pass whose previous read was a cage is a winner
	candidate, every tunnel read is a loser's exit, and the two make an event when the
	winner was inside that tunnel at the exit, left after it, and the follow-through
	falls inside the window. The step computes the same thing with a sweep, a presence
	bitmask and an as-of join, none of which this shares.

	A winner matches at most one of its own passes, since a pass ends before the next
	one starts - which is what the step's as-of join relies on too.

	Args:
		rows: main_df rows, each with ``animal_id``, ``position`` and ``datetime``.
		recording: the recording whose layout names the cages and directional tunnels.
		window: ``AnalysisParams.chasing_time_window``, in seconds.

	Returns:
		One sorted ``(position, winner, loser, winner_exit, chasing_length)`` per event.
	"""
	cages = set(recording.layout.cage_names)
	tunnels = set(recording.layout.tunnel_names_directional)
	shortest, longest = (dt.timedelta(seconds=bound) for bound in window)

	walks: dict[str, list[dict]] = {}
	for row in sorted(rows, key=lambda row: row["datetime"]):
		walks.setdefault(row["animal_id"], []).append(row)

	passes = [
		(current["position"], animal, previous["datetime"], current["datetime"])
		for animal, walk in walks.items()
		for previous, current in pairwise(walk)
		if current["position"] in tunnels and previous["position"] in cages
	]
	exits = [
		(row["position"], row["animal_id"], row["datetime"])
		for row in rows
		if row["position"] in tunnels
	]

	return sorted(
		(position, winner, loser, winner_exit, winner_exit - loser_exit)
		for position, winner, entry, winner_exit in passes
		for tunnel, loser, loser_exit in exits
		if tunnel == position
		and loser != winner
		and entry <= loser_exit < winner_exit
		and shortest < loser_exit - entry < longest
	)
