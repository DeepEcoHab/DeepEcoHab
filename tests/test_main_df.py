"""Tests for build_main_df and build_padded_df (the two tables everything reads).

build_main_df resolves each registration's position from the antenna pair, carries the
last known position on past the final read, and only then trims the lead-in before the
experiment start. build_padded_df cuts the resulting visits at every minute mark and
re-dates each piece by its own start.

Both steps are called directly on a Recording whose ``data`` is a hand-built frame of
raw registrations; padded_df reads main_df via ``Recording.load_results``, which is
monkeypatched as elsewhere.
"""

import datetime as dt

import polars as pl
import pytest
import strategies
from pydantic import ValidationError

from deepecohab.core import recording_pipeline
from deepecohab.core.data_model import AnalysisParams, Boundary, Layout, Recording

at = strategies.at
HALF_DAY = 43200.0  # the default extrapolation_limit, in seconds

# Light onset at midnight, so a window starting at midnight begins its own first phase
# and day/hour come out as calendar day and wall-clock hour.
WINDOW = {"start": "2023-05-24 00:00:00", "finish": "2023-05-26 00:00:00"}
# Acquisition starting at 20:00 puts the nearest light onset 4 h later, so the first 4 h
# of data is recorded but outside the analysed window - the discarded-lead case.
LEAD_WINDOW = {"start": "2023-05-24 20:00:00", "finish": "2023-05-26 00:00:00"}


def recording_with(rows: list[tuple[str, int, dt.datetime]], **overrides) -> Recording:
	"""A recording whose raw data is ``(animal_id, antenna, datetime)`` triples."""
	recording = strategies.analysis_recording(animal_ids=["A", "B"], **{**WINDOW, **overrides})
	recording.data = pl.DataFrame(
		[
			{
				"datetime": moment,
				"antenna": str(antenna),
				"time_under": dt.timedelta(milliseconds=100),
				"animal_id": animal,
				"inserted": False,
			}
			for animal, antenna, moment in rows
		],
		schema=recording.data_schema,
	).lazy()
	return recording


def build(recording: Recording, **params) -> pl.DataFrame:
	"""main_df for ``recording``, sorted per animal as the row order is read below."""
	frame = recording_pipeline.build_main_df(recording, AnalysisParams(**params))
	return frame.collect().sort("animal_id", "datetime")


def rows_for(main: pl.DataFrame, animal: str) -> list[dict]:
	return main.filter(pl.col("animal_id") == animal).to_dicts()


# --- extrapolation -----------------------------------------------------------


def test_extrapolation_is_capped_and_the_rest_is_undefined():
	"""Silence past the limit stops being evidence of staying put.

	The last position is carried for ``extrapolation_limit`` seconds and the remainder
	of the window - a day and a half here - goes to ``undefined`` rather than being
	credited to a cage the animal may have left long before.
	"""
	last_read = at(2023, 5, 24, 0, 15)
	recording = recording_with(
		[("A", 1, at(2023, 5, 24, 0, 5)), ("A", 2, at(2023, 5, 24, 0, 10)), ("A", 2, last_read)]
	)
	_, end = recording.timeline.local_span

	added = rows_for(build(recording), "A")[-2:]

	assert added[0]["datetime"] == last_read + dt.timedelta(seconds=HALF_DAY)
	assert added[0]["time_spent"] == dt.timedelta(seconds=HALF_DAY)
	assert added[0]["position"] == "cage_2"
	assert added[1]["datetime"] == end
	assert added[1]["position"] == Layout.UNDEFINED
	assert added[1]["time_spent"] == end - added[0]["datetime"]


def test_extrapolation_targets_the_window_end_not_the_last_read_in_the_data():
	"""An animal that goes quiet early is followed to the end of the window.

	Targeting the data's last timestamp would leave the stretch after it attributed to
	nobody whenever the animal that goes quiet is not the last one heard from.
	"""
	recording = recording_with([("A", 1, at(2023, 5, 24, 0, 5)), ("B", 1, at(2023, 5, 25, 23, 30))])
	_, end = recording.timeline.local_span

	main = build(recording)

	for animal, first_read in (("A", at(2023, 5, 24, 0, 5)), ("B", at(2023, 5, 25, 23, 30))):
		rows = main.filter(pl.col("animal_id") == animal)
		# Every instant from the animal's first read to the window end is accounted for.
		assert rows["time_spent"].sum() == end - first_read, animal
		assert rows["datetime"].max() == end, animal


def test_cap_row_resolves_to_where_the_animal_is_not_where_it_was():
	"""The carried row repeats the last antenna, so its position is the self-pair.

	The last real read is a tunnel crossing, which names the tunnel the animal came
	*through*. Copying that position would hold it in the tunnel for half a day; reading
	antenna 2 twice resolves to the cage at antenna 2, which is where it ended up.
	"""
	recording = recording_with([("A", 1, at(2023, 5, 24, 0, 5)), ("A", 2, at(2023, 5, 24, 0, 10))])

	last_real, carried = rows_for(build(recording), "A")[-3:-1]

	assert last_real["position"] == "c1_c2"
	assert carried["position"] == "cage_2"
	assert carried["antenna"] == "2"


def test_silence_inside_the_limit_leaves_no_undefined_tail():
	"""When the window ends before the cap does, the carry simply reaches it."""
	recording = recording_with(
		[("A", 1, at(2023, 5, 25, 23, 0)), ("A", 1, at(2023, 5, 25, 23, 30))]
	)
	_, end = recording.timeline.local_span

	added = rows_for(build(recording), "A")[-1]

	assert added["datetime"] == end
	assert added["position"] == "cage_1"
	assert added["time_spent"] == dt.timedelta(minutes=30)


def test_a_real_gap_between_reads_is_never_capped():
	"""The cap applies to the extrapolated tail only.

	Twenty silent hours between two genuine reads are still twenty hours the animal
	spent somewhere the antennas later confirmed; only silence with no read after it is
	guesswork.
	"""
	recording = recording_with([("A", 1, at(2023, 5, 24, 0, 5)), ("A", 1, at(2023, 5, 24, 20, 5))])

	resumed = rows_for(build(recording), "A")[1]

	assert resumed["position"] == "cage_1"
	assert resumed["time_spent"] == dt.timedelta(hours=20)


def test_extrapolation_limit_is_configurable():
	"""The knob moves the cap, and zero disables the carry outright."""
	reads = [("A", 1, at(2023, 5, 24, 0, 5)), ("A", 1, at(2023, 5, 24, 0, 10))]

	hour = rows_for(build(recording_with(reads), extrapolation_limit=3600.0), "A")[-2]
	assert hour["datetime"] == at(2023, 5, 24, 1, 10)

	none = rows_for(build(recording_with(reads), extrapolation_limit=0.0), "A")[-1]
	assert none["position"] == Layout.UNDEFINED
	assert none["time_spent"] == recording_with(reads).timeline.local_span[1] - at(
		2023, 5, 24, 0, 10
	)


# --- recording boundaries ----------------------------------------------------


def stopped(recording: Recording, start: dt.datetime, end: dt.datetime) -> Recording:
	recording.boundaries = [Boundary(start=start, end=end, kind="observed_record_gap")]
	return recording


def test_a_recording_stop_carries_the_position_to_it_then_leaves_it_undefined():
	"""Nothing is known while recording was stopped, nor until the animal is read again.

	The carry ends at the stop rather than running on over it, and the first read after
	the restart - whose antenna pair spans the stop - covers the stretch as undefined.
	"""
	recording = stopped(
		recording_with(
			[
				("A", 1, at(2023, 5, 24, 0, 5)),
				("A", 2, at(2023, 5, 24, 0, 10)),
				("A", 3, at(2023, 5, 24, 6, 0)),
				("A", 3, at(2023, 5, 24, 6, 30)),
			]
		),
		at(2023, 5, 24, 2, 0),
		at(2023, 5, 24, 5, 0),
	)
	_, end = recording.timeline.local_span

	rows = rows_for(build(recording), "A")
	carried, restart, after = rows[2:5]

	assert carried["datetime"] == at(2023, 5, 24, 2, 0)
	assert carried["position"] == "cage_2"
	assert restart["position"] == Layout.UNDEFINED
	assert restart["time_spent"] == dt.timedelta(hours=4)
	assert after["position"] == "cage_3"
	assert sum((row["time_spent"] for row in rows), dt.timedelta()) == end - at(2023, 5, 24, 0, 5)


def test_silence_past_the_limit_before_a_stop_is_undefined_up_to_it():
	"""The cap still applies inside a part: an hour's carry, then undefined to the stop."""
	recording = stopped(
		recording_with([("A", 1, at(2023, 5, 24, 0, 0)), ("A", 1, at(2023, 5, 24, 9, 0))]),
		at(2023, 5, 24, 4, 0),
		at(2023, 5, 24, 8, 0),
	)

	rows = rows_for(build(recording, extrapolation_limit=3600.0), "A")

	assert [(row["datetime"], row["position"]) for row in rows[1:4]] == [
		(at(2023, 5, 24, 1, 0), "cage_1"),
		(at(2023, 5, 24, 4, 0), Layout.UNDEFINED),
		(at(2023, 5, 24, 9, 0), Layout.UNDEFINED),
	]


def test_a_second_read_in_the_same_microsecond_is_dropped():
	"""One animal cannot be read twice at once: the first read in file order is kept.

	It is the first read after the restart, so it has no predecessor and stays undefined.
	"""
	restart = at(2023, 5, 24, 6, 0)
	recording = stopped(
		recording_with([("A", 1, at(2023, 5, 24, 0, 5)), ("A", 1, restart), ("A", 2, restart)]),
		at(2023, 5, 24, 2, 0),
		restart,
	)

	rows = [row for row in rows_for(build(recording), "A") if row["datetime"] == restart]

	assert [(row["antenna"], row["position"]) for row in rows] == [("1", Layout.UNDEFINED)]


def test_a_stop_before_the_window_changes_nothing_inside_it():
	"""The trimmed lead-in absorbs it: in-window rows are as if there had been no stop."""
	reads = [
		("A", 1, at(2023, 5, 24, 20, 30)),
		("A", 2, at(2023, 5, 24, 23, 0)),
		("A", 3, at(2023, 5, 25, 1, 0)),
	]
	plain = build(recording_with(reads, **LEAD_WINDOW))
	after_stop = build(
		stopped(
			recording_with(reads, **LEAD_WINDOW), at(2023, 5, 24, 21, 0), at(2023, 5, 24, 22, 0)
		)
	)

	assert after_stop.equals(plain)


# --- derive, then trim -------------------------------------------------------


def test_the_first_in_window_registration_keeps_its_position():
	"""Positions are resolved before the lead-in is dropped.

	Trimming first leaves the first surviving read with no predecessor to pair with, so
	every animal in a recording with a discarded lead lost its first resolvable
	position. Here the pair 1 -> 2 is formed across the window boundary.
	"""
	recording = recording_with(
		[
			("A", 1, at(2023, 5, 24, 23, 0)),  # discarded lead
			("A", 1, at(2023, 5, 24, 23, 50)),  # discarded lead
			("A", 2, at(2023, 5, 25, 0, 30)),  # first in-window read
		],
		**LEAD_WINDOW,
	)

	first = rows_for(build(recording), "A")[0]

	assert first["datetime"] == at(2023, 5, 25, 0, 30)
	assert first["position"] == "c1_c2"


def test_time_spent_of_the_first_in_window_row_is_clipped_to_the_window():
	"""The visit reaches back before the window, but only the analysed part counts.

	The raw gap is 40 minutes, 10 of which happened before the experiment started.
	Attributing all 40 would credit the window with time it does not cover.
	"""
	recording = recording_with(
		[("A", 1, at(2023, 5, 24, 23, 50)), ("A", 2, at(2023, 5, 25, 0, 30))], **LEAD_WINDOW
	)
	start, _ = recording.timeline.local_span

	first = rows_for(build(recording), "A")[0]

	assert first["time_spent"] == first["datetime"] - start
	assert first["time_spent"] == dt.timedelta(minutes=30)


def test_data_after_the_window_is_dropped_before_extrapolating():
	"""Trailing data is out, and the carry anchors on the window rather than on it."""
	recording = recording_with([("A", 1, at(2023, 5, 25, 23, 0)), ("A", 1, at(2023, 5, 26, 6, 0))])
	_, end = recording.timeline.local_span

	main = build(recording)

	assert main["datetime"].max() == end
	assert main.filter(pl.col("datetime") > end).height == 0


# --- position resolution -----------------------------------------------------


def test_unmapped_antenna_pair_is_undefined():
	"""A jump the layout cannot join - antennas the animal passed unread - is no place."""
	recording = recording_with([("A", 1, at(2023, 5, 24, 0, 5)), ("A", 4, at(2023, 5, 24, 0, 10))])

	assert rows_for(build(recording), "A")[1]["position"] == Layout.UNDEFINED


def test_first_read_sentinel_does_not_collide_with_an_antenna_numbered_zero():
	"""The unpaired first read is filled with a non-numeric sentinel, not with 0.

	In a layout that numbers an antenna 0, filling the missing predecessor with 0 would
	make the very first read look like the self-pair "0_0" and place the animal in a
	cage it has not been seen in yet.
	"""
	recording = strategies.analysis_recording(
		animal_ids=["A", "B"],
		antenna_combinations={
			"0_0": "cage_1",
			"1_1": "cage_2",
			"0_1": "c1_c2",
			"1_0": "c2_c1",
		},
		tunnels_map={"c1_c2": "tunnel_1", "c2_c1": "tunnel_1"},
		**WINDOW,
	)
	recording.data = recording_with(
		[("A", 0, at(2023, 5, 24, 0, 5)), ("A", 0, at(2023, 5, 24, 0, 10))]
	).data

	positions = [row["position"] for row in rows_for(build(recording), "A")]

	assert positions[0] == Layout.UNDEFINED
	assert positions[1] == "cage_1"


def test_no_registration_leaves_without_a_phase_count():
	"""Every row, added ones included, lands in a bin of the dense grid.

	The carried and tail rows sit exactly on the window end, which is the one instant a
	grid built over a half-open window would not cover.
	"""
	recording = recording_with(
		[
			("A", 1, at(2023, 5, 24, 11, 59)),
			("A", 2, at(2023, 5, 24, 12, 1)),  # crosses the light -> dark boundary
			("B", 1, at(2023, 5, 25, 23, 59)),
		]
	)

	main = build(recording)

	calendar = main.select(pl.col("phase", "day", "hour", "phase_count"))
	assert calendar.null_count().sum_horizontal().item() == 0
	# The carried rows sit on the window end, in the bin that instant opens.
	assert main.filter(pl.col("datetime") == recording.timeline.local_span[1]).height == 2


# --- padded_df ---------------------------------------------------------------


def padded(monkeypatch, recording: Recording, main: pl.LazyFrame | pl.DataFrame) -> pl.DataFrame:
	monkeypatch.setattr(Recording, "load_results", lambda self, key, eager=False: main.lazy())
	return recording_pipeline.build_padded_df(recording, AnalysisParams()).collect()


def test_phase_count_is_re_derived_from_the_piece_start(monkeypatch):
	"""A piece belongs to the phase it began in, not to the one its visit ended in.

	The visit straddles the light -> dark boundary, so it arrives carrying the dark
	phase's number. Its first piece happened entirely in the light phase and has to say
	so, or the last minute of every phase is credited to the next one.
	"""
	recording = recording_with([("A", 1, at(2023, 5, 24, 11, 59))])
	main = build(recording)
	visit = pl.DataFrame(
		main.rows(named=True)[:1],
		schema=main.schema,
	).with_columns(
		pl.lit(at(2023, 5, 24, 12, 0, 30)).alias("datetime"),
		pl.lit(dt.timedelta(seconds=60)).alias("time_spent"),
		pl.lit("dark_phase", dtype=main.schema["phase"]).alias("phase"),
		pl.lit(12, dtype=pl.UInt8).alias("hour"),
		pl.lit(2, dtype=pl.UInt16).alias("phase_count"),
	)

	pieces = padded(monkeypatch, recording, visit).sort("datetime")

	assert pieces.height == 2
	assert pieces["hour"].to_list() == [11, 12]
	assert pieces["phase"].to_list() == ["light_phase", "dark_phase"]
	assert pieces["phase_count"].to_list() == [1, 2]
	assert pieces["time_spent"].to_list() == [dt.timedelta(seconds=30)] * 2


def test_pieces_tile_the_visit_exactly(monkeypatch):
	"""Splitting conserves time: the pieces sum to the visit they came from."""
	recording = recording_with(
		[("A", 1, at(2023, 5, 24, 0, 0, 20)), ("A", 1, at(2023, 5, 24, 0, 3, 40))]
	)
	main = build(recording)

	pieces = padded(monkeypatch, recording, main)

	assert pieces["time_spent"].sum() == main["time_spent"].sum()
	assert (~pieces["interpolated"]).sum() == main.height  # one first piece per visit


def test_row_order_and_row_id_do_not_depend_on_input_order(monkeypatch):
	"""Reruns write identical tables, row for row.

	The two animals read at the same instants and are both carried to the window end,
	so ``datetime`` alone leaves ties. padded_df's ``row_id`` is main_df's row index, so
	an unpinned tie order renumbers the visits on every run.
	"""
	rows = [
		("A", 1, at(2023, 5, 24, 0, 5)),
		("B", 2, at(2023, 5, 24, 0, 5)),
		("A", 2, at(2023, 5, 24, 0, 10)),
		("B", 1, at(2023, 5, 24, 0, 10)),
	]
	recordings = [recording_with(order) for order in (rows, rows[::-1])]
	mains = [
		recording_pipeline.build_main_df(recording, AnalysisParams()).collect()
		for recording in recordings
	]
	paddeds = [padded(monkeypatch, *pair) for pair in zip(recordings, mains, strict=True)]

	assert mains[0].equals(mains[1])
	assert paddeds[0].equals(paddeds[1])


def test_no_piece_starts_before_the_analysed_window(monkeypatch):
	"""The clipped first visit is what keeps padding inside the grid.

	A piece starting before the experiment start would carry a day and hour the grid
	does not hold, so it would leave with a null phase_count and be dropped silently at
	the reindex - taking its visit's only non-interpolated flag with it.
	"""
	recording = recording_with(
		[("A", 1, at(2023, 5, 24, 23, 50)), ("A", 2, at(2023, 5, 25, 0, 30))], **LEAD_WINDOW
	)
	start, _ = recording.timeline.local_span
	main = build(recording)

	pieces = padded(monkeypatch, recording, main)

	assert (pieces["datetime"] - pieces["time_spent"]).min() == start
	assert pieces["phase_count"].null_count() == 0


# --- the antennas the layout names -------------------------------------------
# recording_with() attaches its frame after validation, so these go through
# model_validate to reach the check a real config load runs.


def loaded_with(antennas: list[str]) -> Recording:
	"""A recording whose data reads ``antennas``, validated as a config load would."""
	recording = strategies.analysis_recording(animal_ids=["A"], **WINDOW)
	frame = pl.LazyFrame(
		{
			"datetime": [at(2023, 5, 24, 0, index) for index in range(len(antennas))],
			"antenna": antennas,
			"time_under": [dt.timedelta(milliseconds=100)] * len(antennas),
			"animal_id": ["A"] * len(antennas),
			"inserted": [False] * len(antennas),
		},
		schema=recording.data_schema,
	)
	return Recording.model_validate({**recording.model_dump(mode="json"), "data": frame})


def test_antenna_the_layout_does_not_name_is_rejected():
	"""Data and layout naming antennas differently would read as every position undefined."""
	with pytest.raises(ValidationError, match="does not name"):
		loaded_with(["1", "9"])


def test_antenna_that_never_read_is_not_an_error():
	"""A dead antenna is a quality finding, not a broken config."""
	assert loaded_with(["1", "2"]).data.collect().height == 2


def test_optional_column_is_accepted_when_present_and_typed():
	"""internal_board_timestamp may be absent, but when present it must match its dtype."""
	recording = loaded_with(["1"])
	stamps = recording.data.with_columns(internal_board_timestamp=pl.col("datetime"))
	Recording.model_validate({**recording.model_dump(mode="json"), "data": stamps})

	untyped = stamps.with_columns(pl.col("internal_board_timestamp").cast(pl.Int64))
	with pytest.raises(
		ValidationError, match="internal_board_timestamp is Int64, expected Datetime"
	):
		Recording.model_validate({**recording.model_dump(mode="json"), "data": untyped})


@pytest.mark.parametrize(
	("change", "problem"),
	[
		(lambda df: df.drop("antenna"), "no antenna column"),
		(lambda df: df.drop("inserted"), "no inserted column"),
		(lambda df: df.with_columns(extra=pl.lit(1)), "unexpected extra column"),
		(
			lambda df: df.select(reversed(df.collect_schema().names())),
			"columns must be in the order datetime, antenna, time_under, animal_id, inserted",
		),
	],
)
def test_schema_mismatch_names_each_difference(change, problem):
	recording = loaded_with(["1"])
	with pytest.raises(ValidationError, match=f"schema mismatch - {problem}"):
		Recording.model_validate(
			{**recording.model_dump(mode="json"), "data": change(recording.data)}
		)
