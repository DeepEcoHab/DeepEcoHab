import datetime as dt
from functools import partial

import polars as pl
import pytest
import strategies
from hypothesis import given, settings, strategies as st

from deepecohab.core import grids, transforms
from deepecohab.core.data_model import Recording

TZ_NAME = "Europe/Warsaw"
# Same zone as the frames make_frame builds: day and hour are now measured from the
# recording's own origin, so a frame has to belong to the recording it is split for.
# Its phases switch on the hour, so every cell is one whole hour.
RECORDING = strategies.analysis_recording(
	tz=TZ_NAME, start="2023-01-01 00:00:00", finish="2023-12-31 23:00:00"
)


def _recording(tz: str, phases: dict) -> Recording:
	"""A recording spanning the whole generated year, in the drawn zone and phases."""
	return strategies.analysis_recording(
		tz=tz, start="2023-01-01 00:00:00", finish="2023-12-31 23:00:00", phases=phases
	)


MINUTE = dt.timedelta(minutes=1)


at = partial(strategies.at, tz=TZ_NAME)


def make_frame(rows: list[dict]) -> pl.LazyFrame:
	"""rows: {"end": datetime, "time_spent": seconds, "time_under": timedelta,
	optional "animal_id"/"position"}. Cases state time_spent in seconds; the frame
	carries it as a duration, as main_df does.
	"""
	return pl.LazyFrame(
		{
			"animal_id": pl.Series(
				[r.get("animal_id", "A") for r in rows], dtype=pl.Enum(["A", "B"])
			),
			"position": pl.Series(
				[r.get("position", "cage_1") for r in rows], dtype=pl.Categorical
			),
			"datetime": pl.Series(
				"datetime", [r["end"] for r in rows], dtype=pl.Datetime("us", TZ_NAME)
			),
			"time_spent": strategies.seconds([float(r["time_spent"]) for r in rows]),
			"time_under": pl.Series([r["time_under"] for r in rows], dtype=pl.Duration("us")),
		}
	)


def split(lf: pl.LazyFrame, recording=None) -> pl.DataFrame:
	"""Cell-split pieces, with time_spent read back in seconds as the cases state it."""
	return (
		transforms.split_visits_on_cells(lf, recording or RECORDING)
		.with_columns(pl.col("time_spent").dt.total_seconds(fractional=True))
		.collect()
	)


def crosses_a_cell_start(out: pl.DataFrame, recording: Recording) -> bool:
	"""Whether any piece has a cell of ``recording`` beginning strictly inside it."""
	starts = out["datetime"] - pl.Series(
		[dt.timedelta(seconds=s) for s in out["time_spent"]], dtype=pl.Duration("us")
	)
	cells = grids.cell_starts(recording)
	before_end = cells.search_sorted(out["datetime"], side="left").cast(pl.Int64)
	inside = before_end - cells.search_sorted(starts, side="right").cast(pl.Int64)
	return bool((inside > 0).any())


def test_within_cell_not_split():
	"""Interval inside one hour -> single passthrough row."""
	lf = make_frame(
		[{"end": at(2023, 6, 15, 12, 0, 40), "time_spent": 20, "time_under": MINUTE / 4}]
	)  # 12:00:20 -> 12:00:40
	out = split(lf)
	assert out.height == 1
	assert out["interpolated"].to_list() == [False]
	assert out["time_spent"][0] == pytest.approx(20)
	assert out["time_under"][0] == MINUTE / 4


def test_minute_mark_inside_a_cell_does_not_split():
	"""12:00:40 -> 12:01:20 crosses a minute mark but no cell start -> one piece."""
	lf = make_frame([{"end": at(2023, 6, 15, 12, 1, 20), "time_spent": 40, "time_under": MINUTE}])
	out = split(lf)
	assert out.height == 1
	assert out["time_spent"][0] == pytest.approx(40)


def test_interval_exactly_one_aligned_cell_not_split():
	"""[12:00:00, 13:00:00) is one whole cell -> single piece."""
	lf = make_frame([{"end": at(2023, 6, 15, 13, 0, 0), "time_spent": 3600, "time_under": MINUTE}])
	out = split(lf)
	assert out.height == 1
	assert out["interpolated"].to_list() == [False]


def test_single_cell_crossing_splits_into_two():
	"""11:59:40 -> 12:00:20 crosses 12:00 -> two 20s pieces; first kept, second interpolated."""
	lf = make_frame([{"end": at(2023, 6, 15, 12, 0, 20), "time_spent": 40, "time_under": MINUTE}])
	out = split(lf).sort("datetime")
	assert out.height == 2
	assert out["interpolated"].to_list() == [False, True]
	assert out["time_spent"].to_list() == pytest.approx([20, 20])
	halves = out["time_under"].to_list()
	assert halves[0] == pytest.approx(MINUTE / 2, abs=dt.timedelta(microseconds=2))
	assert halves[1] == pytest.approx(MINUTE / 2, abs=dt.timedelta(microseconds=2))


def test_multi_cell_crossing_piece_count():
	"""10:59:40 -> 13:00:20 crosses three cell starts and yields four pieces."""
	lf = make_frame(
		[{"end": at(2023, 6, 15, 13, 0, 20), "time_spent": 2 * 3600 + 40, "time_under": MINUTE}]
	)
	out = split(lf).sort("datetime")
	assert out["time_spent"].to_list() == pytest.approx([20, 3600, 3600, 20])
	# only the first piece represents the visit; the rest are interpolated
	assert out["interpolated"].to_list() == [False, True, True, True]


def test_time_spent_conserved():
	original = 40
	lf = make_frame(
		[{"end": at(2023, 6, 15, 12, 0, 20), "time_spent": original, "time_under": MINUTE}]
	)
	out = split(lf)
	assert out["time_spent"].sum() == pytest.approx(original)


def test_time_under_conserved():
	tu = dt.timedelta(seconds=37)
	lf = make_frame([{"end": at(2023, 6, 15, 12, 0, 20), "time_spent": 40, "time_under": tu}])
	out = split(lf)
	assert out["time_under"].sum() == pytest.approx(tu, abs=dt.timedelta(microseconds=4))


def test_no_piece_crosses_a_cell_start():
	lf = make_frame(
		[{"end": at(2023, 6, 15, 13, 0, 20), "time_spent": 2 * 3600 + 40, "time_under": MINUTE}]
	)
	assert not crosses_a_cell_start(split(lf), RECORDING)


def test_hour_label_from_piece_start():
	"""Hour column reflects the piece START, not the original interval end."""
	# 11:59:40 -> 12:00:20 crosses the hour mark at 12:00.
	lf = make_frame([{"end": at(2023, 6, 15, 12, 0, 20), "time_spent": 40, "time_under": MINUTE}])
	# An origin in the same DST regime as the data, so egocentric hours and the wall
	# clock agree and this is about the piece start alone. The DST case is its own test.
	recording = strategies.analysis_recording(
		tz=TZ_NAME, start="2023-06-15 00:00:00", finish="2023-06-16 00:00:00"
	)
	out = split(lf, recording).sort("datetime")
	assert out.height == 2
	# first piece 11:59:40-12:00:00 -> hour 11; second 12:00:00-12:00:20 -> hour 12
	assert out["hour"].cast(pl.Int64).to_list() == [11, 12]


def test_phase_switch_inside_an_hour_splits_it():
	"""With dark from 19:20, the 19:00 hour holds two cells, and a visit across 19:20 is cut."""
	recording = strategies.analysis_recording(
		tz=TZ_NAME,
		start="2023-06-15 00:00:00",
		finish="2023-06-17 00:00:00",
		phases={"light_phase": dt.time(7, 0), "dark_phase": dt.time(19, 20)},
	)
	lf = make_frame([{"end": at(2023, 6, 15, 19, 30, 0), "time_spent": 1200, "time_under": MINUTE}])
	out = split(lf, recording).sort("datetime")
	assert out["time_spent"].to_list() == pytest.approx([600, 600])
	assert out["phase"].cast(pl.String).to_list() == ["light_phase", "dark_phase"]
	assert out["hour"].n_unique() == 1


def test_zero_duration_row_survives_without_nan():
	lf = make_frame(
		[{"end": at(2023, 6, 15, 12, 0, 0), "time_spent": 0, "time_under": dt.timedelta(0)}]
	)
	out = split(lf)
	assert out.height == 1
	assert out["interpolated"].to_list() == [False]
	assert out["time_under"][0] == dt.timedelta(0)


def test_row_id_present_in_output():
	lf = make_frame([{"end": at(2023, 6, 15, 12, 0, 20), "time_spent": 40, "time_under": MINUTE}])
	out = split(lf)
	assert "row_id" in out.columns


def test_row_id_dedup_recovers_original_count():
	"""unique(row_id, keep=first) returns exactly one row per original visit."""
	rows = [
		{"end": at(2023, 6, 15, 12, 0, 40), "time_spent": 20, "time_under": MINUTE},  # no split
		{"end": at(2023, 6, 15, 12, 0, 20), "time_spent": 40, "time_under": MINUTE},  # 2 pieces
		{"end": at(2023, 6, 15, 13, 0, 20), "time_spent": 7240, "time_under": MINUTE},  # 4 pieces
	]
	lf = make_frame(rows)
	out = split(lf)

	# more pieces than originals, but exactly len(rows) distinct row_ids
	assert out.height > len(rows)
	assert out["row_id"].n_unique() == len(rows)
	deduped = out.unique("row_id", keep="first")
	assert deduped.height == len(rows)
	assert sorted(deduped["row_id"].to_list()) == list(range(len(rows)))


def test_row_ids_are_contiguous_from_zero():
	rows = [
		{"end": at(2023, 6, 15, 12, 0, 40), "time_spent": 20, "time_under": MINUTE}
		for _ in range(5)
	]
	lf = make_frame(rows)
	out = split(lf)
	assert sorted(out["row_id"].unique().to_list()) == list(range(5))


def test_dedup_first_piece_holds_piece_values_not_original():
	"""Documents that keep='first' yields the FIRST PIECE's durations.

	For a split visit, the deduped row's time_spent is the first piece's length,
	NOT the original visit length. This is correct for counting but must not be
	read as the original duration.
	"""
	lf = make_frame(
		[{"end": at(2023, 6, 15, 12, 0, 20), "time_spent": 40, "time_under": MINUTE}]
	)  # splits into 20s + 20s at 12:00:00
	out = split(lf)
	first = out.unique("row_id", keep="first").row(0, named=True)
	assert first["time_spent"] == pytest.approx(20)  # first piece, not 40


def test_non_interpolated_filter_recovers_original_count_per_animal():
	"""Filtering interpolated==False yields the same row count per animal as the input.

	Each original visit contributes exactly one non-interpolated piece regardless of
	how many cells it crosses, so padded_df.filter(interpolated==False) matches
	main_df row-for-row per animal.
	"""
	rows = [
		{
			"animal_id": "A",
			"end": at(2023, 6, 15, 12, 0, 40),
			"time_spent": 20,
			"time_under": MINUTE,
		},  # no split
		{
			"animal_id": "A",
			"end": at(2023, 6, 15, 12, 0, 20),
			"time_spent": 40,
			"time_under": MINUTE,
		},  # 2 pieces
		{
			"animal_id": "B",
			"end": at(2023, 6, 15, 13, 0, 20),
			"time_spent": 7240,
			"time_under": MINUTE,
		},  # 4 pieces
	]
	out = split(make_frame(rows))
	kept = out.filter(~pl.col("interpolated"))

	assert kept.height == len(rows)  # more pieces overall, but one kept row per visit
	per_animal = dict(kept.group_by("animal_id").len().sort("animal_id").iter_rows())
	assert per_animal == {"A": 2, "B": 1}


# --- property-based tests ----------------------------------------------------
# Timezone and phase config are drawn inside @given (st.sampled_from), so each
# example exercises a random combination, including DST zones and midnight
# phase boundaries; see tests/strategies.py.


@settings(max_examples=200, deadline=None)
@given(v=strategies.visit, tz=strategies.timezones, pcfg=strategies.phase_configs)
def test_property_conserves_time_spent(v, tz, pcfg):
	"""However a single visit is split across cells, total time_spent is preserved and
	no piece crosses the start of a cell, in any timezone / phase config.
	"""
	recording = _recording(tz, pcfg)
	out = split(strategies.padding_frame([v], tz), recording)
	assert out["time_spent"].sum() == pytest.approx(v["time_spent"], abs=1e-6)
	assert not crosses_a_cell_start(out, recording)


@settings(max_examples=200, deadline=None)
@given(
	rows=st.lists(strategies.visit, min_size=1, max_size=20),
	tz=strategies.timezones,
	pcfg=strategies.phase_configs,
)
def test_property_preserves_identity_and_dedup(rows, tz, pcfg):
	"""Padding may explode rows into per-cell pieces, but unique(row_id,
	keep="first") recovers exactly one row per original visit, filtering
	interpolated==False keeps exactly one piece per visit, and every piece keeps the
	animal_id/position of the visit it came from (row_id i == row i).
	"""
	out = transforms.split_visits_on_cells(
		strategies.padding_frame(rows, tz), _recording(tz, pcfg)
	).collect()

	assert out["row_id"].n_unique() == len(rows)
	assert sorted(out["row_id"].unique().to_list()) == list(range(len(rows)))
	assert out.unique("row_id", keep="first").height == len(rows)
	assert out.filter(~pl.col("interpolated")).height == len(rows)

	for i, r in enumerate(rows):
		pieces = out.filter(pl.col("row_id") == i)
		assert pieces["animal_id"].unique().to_list() == [r["animal_id"]]
		assert pieces["position"].unique().to_list() == [r["position"]]
