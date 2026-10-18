import datetime as dt
from functools import partial
from zoneinfo import ZoneInfo

import polars as pl
import pytest
import strategies
from hypothesis import given, settings, strategies as st

from deepecohab.core.antenna_analysis import _get_time_alone
from deepecohab.core.data_model import AnalysisParams

TZ = ZoneInfo("Europe/Warsaw")

SCHEMA: dict[str, pl.DataType] = {
	"animal_id": pl.Enum(["A", "B", "C"]),
	"position": pl.Categorical,
	"datetime": pl.Datetime("us", str(TZ)),
	"time_spent": pl.Duration("us"),
}

# The span covers every datetime used by the example-based tests below; the grid it
# builds is what _get_time_alone looks phase_count up from.
RECORDING = strategies.analysis_recording(
	animal_ids=["A", "B", "C"],
	tz=str(TZ),
	start="2023-05-24 00:00:00",
	finish="2023-05-26 23:00:00",
)


def _recording_spanning(frame: pl.DataFrame, tz: str, phases=None):
	"""A recording whose window spans *frame*.

	The property tests draw datetimes from across the year, so the grid window is
	derived from the data itself rather than hard-coded.
	"""
	# Padded a day either side so the window is non-empty even for a single event,
	# and so the grid covers the hour every event falls in.
	first = frame["datetime"].min().replace(tzinfo=None) - dt.timedelta(days=1)
	last = frame["datetime"].max().replace(tzinfo=None) + dt.timedelta(days=1)
	return strategies.analysis_recording(
		animal_ids=["A", "B", "C"], tz=tz, start=str(first), finish=str(last), phases=phases
	)


EXPECTED_COLUMNS = {
	"animal_id",
	"position",
	"phase",
	"day",
	"phase_count",
	"hour",
	"time_alone",
}


at = partial(strategies.at, tz=str(TZ))


@pytest.fixture
def make_df():
	"""Build a typed, zone-aware DataFrame from plain column dicts."""

	def _make(data: dict) -> pl.DataFrame:
		# Cases state time_spent in seconds, which reads better than timedeltas.
		data = {**data, "time_spent": [dt.timedelta(seconds=v) for v in data["time_spent"]]}
		return pl.DataFrame(data, schema=SCHEMA)

	return _make


def alone(frame: pl.DataFrame, recording=None, minimum_time_alone: float = 0.0) -> pl.DataFrame:
	"""Solitary time, with time_alone read back in seconds as the cases state it.

	These cases exercise the sweep, so the threshold is off unless a case sets it.
	"""
	params = AnalysisParams(minimum_time_alone=minimum_time_alone)
	return (
		_get_time_alone(frame.lazy(), recording or RECORDING, params)
		.with_columns(pl.col("time_alone").dt.total_seconds(fractional=True))
		.collect()
	)


def alone_row(result: pl.DataFrame, animal: str) -> dict:
	"""Extract the single result row for *animal*; fail clearly otherwise."""
	rows = result.filter(pl.col("animal_id") == animal)
	assert rows.height == 1, f"Expected exactly one row for animal '{animal}', got {rows.height}"
	return rows.row(0, named=True)


def test_output_schema(make_df):
	"""Result carries animal_id, position, phase, day, time_alone."""
	df = make_df(
		{
			"animal_id": ["A"],
			"position": ["cage_1"],
			"datetime": [dt.datetime(2023, 5, 24, 12, 0, 0, tzinfo=TZ)],
			"time_spent": [10.0],
		}
	)
	result = alone(df)
	assert set(result.columns) == EXPECTED_COLUMNS


def test_threshold_applies_to_the_whole_bout(make_df):
	"""Back-to-back reads of one solitary bout count together against minimum_time_alone.

	8 s then 7 s in the same cage is one 15 s bout, kept whole at a 10 s threshold even
	though neither read clears it alone; a separate 6 s bout is still dropped.
	"""
	df = make_df(
		{
			"animal_id": ["A", "A", "A"],
			"position": ["cage_1", "cage_1", "cage_1"],
			"datetime": [
				at(2023, 5, 24, 12, 0, 8),
				at(2023, 5, 24, 12, 0, 15),
				at(2023, 5, 24, 12, 5, 6),
			],
			"time_spent": [8.0, 7.0, 6.0],
		}
	)
	assert alone_row(alone(df, minimum_time_alone=10.0), "A")["time_alone"] == 15


def test_partial_overlap(make_df):
	"""Staggered overlap, each animal alone for 5 s:
	A 12:00:00-12:00:10, B 12:00:05-12:00:15, together 12:00:05-12:00:10.
	"""
	df = make_df(
		{
			"animal_id": ["A", "B"],
			"position": ["cage_1", "cage_1"],
			"datetime": [
				dt.datetime(2023, 5, 24, 12, 0, 0, tzinfo=TZ),
				dt.datetime(2023, 5, 24, 12, 0, 5, tzinfo=TZ),
			],
			"time_spent": [10.0, 10.0],
		}
	)
	result = alone(df)

	assert result.height == 2
	assert alone_row(result, "A")["time_alone"] == 5
	assert alone_row(result, "B")["time_alone"] == 5


def test_empty_gap_not_counted(make_df):
	"""A gap between visits must not be credited to either animal (5 s each)."""
	df = make_df(
		{
			"animal_id": ["A", "B"],
			"position": ["cage_1", "cage_1"],
			"datetime": [
				dt.datetime(2023, 5, 24, 12, 0, 0, tzinfo=TZ),
				dt.datetime(2023, 5, 24, 12, 0, 20, tzinfo=TZ),
			],
			"time_spent": [5.0, 5.0],
		}
	)
	result = alone(df)

	assert result.height == 2
	assert alone_row(result, "A")["time_alone"] == 5
	assert alone_row(result, "B")["time_alone"] == 5


def test_multiple_positions_independent(make_df):
	"""Alone time is per position; cage_2 presence does not affect cage_1."""
	df = make_df(
		{
			"animal_id": ["A", "A", "B"],
			"position": ["cage_1", "cage_2", "cage_1"],
			"datetime": [
				dt.datetime(2023, 5, 24, 12, 0, 0, tzinfo=TZ),
				dt.datetime(2023, 5, 24, 12, 0, 0, tzinfo=TZ),
				dt.datetime(2023, 5, 24, 12, 0, 0, tzinfo=TZ),
			],
			"time_spent": [10.0, 10.0, 10.0],
		}
	)
	result = alone(df)

	assert result.filter(pl.col("position") == "cage_2").height == 1
	assert result.filter(pl.col("position") == "cage_1").height == 0
	cage2 = result.filter((pl.col("animal_id") == "A") & (pl.col("position") == "cage_2")).row(
		0, named=True
	)
	assert cage2["time_alone"] == 10


def test_accumulated_alone_time_same_phase(make_df):
	"""Separate non-overlapping visits in the same phase/day are summed."""
	df = make_df(
		{
			"animal_id": ["A", "A"],
			"position": ["cage_1", "cage_1"],
			"datetime": [
				dt.datetime(2023, 5, 24, 12, 0, 10, tzinfo=TZ),
				dt.datetime(2023, 5, 24, 12, 1, 10, tzinfo=TZ),
			],
			"time_spent": [10.0, 10.0],
		}
	)
	result = alone(df)

	assert result.height == 1
	assert alone_row(result, "A")["time_alone"] == 20


def test_phase_split(make_df):
	"""Two alone visits in different phases produce two rows, labeled correctly."""
	df = make_df(
		{
			"animal_id": ["A", "A"],
			"position": ["cage_1", "cage_1"],
			"datetime": [
				dt.datetime(2023, 5, 24, 12, 0, 0, tzinfo=TZ),  # 12:00 -> light_phase
				dt.datetime(2023, 5, 24, 22, 0, 0, tzinfo=TZ),  # 22:00 -> dark_phase
			],
			"time_spent": [10.0, 10.0],
		}
	)
	result = alone(df).sort("phase")

	assert result.height == 2
	by_phase = {r["phase"]: r for r in result.iter_rows(named=True)}
	assert set(by_phase) == {"light_phase", "dark_phase"}
	assert by_phase["light_phase"]["time_alone"] == 10
	assert by_phase["dark_phase"]["time_alone"] == 10


def test_day_split(make_df):
	"""Visits on different calendar days produce separate, 1-indexed day rows."""
	df = make_df(
		{
			"animal_id": ["A", "A"],
			"position": ["cage_1", "cage_1"],
			"datetime": [
				dt.datetime(2023, 5, 24, 12, 0, 0, tzinfo=TZ),  # 1st day
				dt.datetime(2023, 5, 26, 12, 0, 0, tzinfo=TZ),  # 3rd day
			],
			"time_spent": [10.0, 10.0],
		}
	)
	result = alone(df).sort("day")

	assert result.height == 2
	assert result["day"].to_list() == [1, 3]
	assert all(t == 10 for t in result["time_alone"])


def test_simultaneous_handoff(make_df):
	"""A leaves at T and B enters at T: each animal gets its full alone time, no double-count."""
	# A: [12:00:00, 12:00:10), B: [12:00:10, 12:00:20) — leave and enter at the same instant.
	df = make_df(
		{
			"animal_id": ["A", "B"],
			"position": ["cage_1", "cage_1"],
			"datetime": [at(2023, 5, 24, 12, 0, 10), at(2023, 5, 24, 12, 0, 20)],
			"time_spent": [10.0, 10.0],
		}
	)
	result = alone(df)

	assert result.height == 2
	assert alone_row(result, "A")["time_alone"] == 10
	assert alone_row(result, "B")["time_alone"] == 10


def test_simultaneous_swap_with_bystander(make_df):
	"""A leaves and C enters at T while B is present throughout — nobody is ever alone."""
	# A: [12:00:00, 12:00:10), B: [12:00:00, 12:00:20), C: [12:00:10, 12:00:20)
	df = make_df(
		{
			"animal_id": ["A", "B", "C"],
			"position": ["cage_1", "cage_1", "cage_1"],
			"datetime": [
				at(2023, 5, 24, 12, 0, 10),
				at(2023, 5, 24, 12, 0, 20),
				at(2023, 5, 24, 12, 0, 20),
			],
			"time_spent": [10.0, 20.0, 10.0],
		}
	)
	result = alone(df)

	assert result.height == 0


# --- property-based tests ----------------------------------------------------
# A full alone-time oracle would just reimplement the function, so these assert
# structural invariants that must hold for any arrangement of animals/intervals.
# Timezone and phase config are drawn inside @given; see tests/strategies.py.
# tunnels={} keeps positions undirected so remove_tunnel_directionality is a
# no-op and cage positions pass straight through.


@settings(max_examples=200)
@given(
	events=st.lists(strategies.event, min_size=1, max_size=25),
	tz=strategies.timezones,
	pcfg=strategies.phase_configs,
)
def test_alone_never_exceeds_presence(events, tz, pcfg):
	"""No animal can be alone longer than the total time it is present: summed
	time_alone per animal <= summed visit duration per animal.
	"""
	frame = strategies.time_alone_frame(events, tz)
	recording = _recording_spanning(frame, tz, phases=pcfg)
	result = alone(frame, recording)

	totals = dict(result.group_by("animal_id").agg(pl.sum("time_alone")).iter_rows())
	# Durations reach the pipeline quantised to whole microseconds, so the reference
	# quantises the same way rather than summing the raw floats.
	presence: dict[str, float] = {}
	for e in events:
		quantised = dt.timedelta(seconds=e["duration"]).total_seconds()
		presence[e["animal_id"]] = presence.get(e["animal_id"], 0.0) + quantised

	assert (result["time_alone"] >= 0).all()
	for animal, total_alone in totals.items():
		assert total_alone <= presence[str(animal)] + 1e-6


@settings(max_examples=200)
@given(
	animals=st.lists(st.sampled_from(strategies.ANIMALS), min_size=2, max_size=6, unique=True),
	position=st.sampled_from(strategies.CAGES),
	start=strategies.naive_datetimes,
	duration=strategies.positive_durations,
	tz=strategies.timezones,
	pcfg=strategies.phase_configs,
)
def test_full_overlap_means_nobody_alone(animals, position, start, duration, tz, pcfg):
	"""When >=2 animals occupy the exact same interval in one cage, at no instant
	is exactly one present, so the result is empty.
	"""
	events = [
		{"animal_id": a, "position": position, "start": start, "duration": duration}
		for a in animals
	]
	frame = strategies.time_alone_frame(events, tz)
	recording = _recording_spanning(frame, tz, phases=pcfg)
	result = alone(frame, recording)
	assert result.height == 0


@settings(max_examples=200)
@given(
	animal=st.sampled_from(strategies.ANIMALS),
	position=st.sampled_from(strategies.CAGES),
	start=strategies.naive_datetimes,
	duration=strategies.positive_durations,
	tz=strategies.timezones,
	pcfg=strategies.phase_configs,
)
def test_lone_animal_is_alone_for_its_whole_visit(animal, position, start, duration, tz, pcfg):
	"""A single animal with a single visit is alone for exactly that duration."""
	events = [{"animal_id": animal, "position": position, "start": start, "duration": duration}]
	frame = strategies.time_alone_frame(events, tz)
	recording = _recording_spanning(frame, tz, phases=pcfg)
	result = alone(frame, recording)

	assert result.height == 1
	assert result["animal_id"].to_list() == [animal]
	assert result["time_alone"][0] == pytest.approx(duration, abs=1e-6)
