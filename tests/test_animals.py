"""Tests for the animals table.

The step reshapes cohort metadata into a frame joinable to any analysis table, so
the assertions here are about the join key, the recorded fields surviving intact,
and the one derived column (age at the start of the recording).
"""

import datetime as dt

import polars as pl
import strategies

from deepecohab.core import recording_pipeline
from deepecohab.core.data_model import AnalysisParams

RECORDING = strategies.analysis_recording(animal_ids=["A", "B", "C"], start="2023-05-24 00:00:00")


def build(recording=None) -> pl.DataFrame:
	return recording_pipeline.build_animals(recording or RECORDING, AnalysisParams()).collect()


def test_one_row_per_cohort_animal():
	animals = build()
	assert animals.height == RECORDING.cohort.n_mice
	assert animals["animal_id"].to_list() == RECORDING.cohort.animal_tags


def test_carries_every_recorded_field():
	expected = {
		"animal_id",
		"subject_name",
		"mouse_line",
		"genotype",
		"sex",
		"date_of_birth",
		"age",
		"genetic_background",
		"treatment",
		"notes",
	}
	assert set(build().columns) == expected


def test_join_key_matches_the_analysis_tables():
	"""animal_id shares the enum the pipeline uses, so a join needs no casting."""
	animals = build()
	padded = strategies.padded_df_frame(
		[
			{
				"animal_id": "A",
				"position": "cage_1",
				"datetime": strategies.at(2023, 5, 24, 12, 0, 0),
				"time_spent": 10,
			}
		],
		RECORDING,
	).collect()

	assert animals.schema["animal_id"] == padded.schema["animal_id"]

	joined = padded.join(animals, on="animal_id", how="left")
	assert joined.height == padded.height
	assert joined["genotype"].null_count() == 0


def test_age_is_whole_days_at_recording_start():
	animals = build()
	start = RECORDING.timeline.local_span[0].date()

	for row in animals.iter_rows(named=True):
		assert row["age"] == start - row["date_of_birth"]
		assert row["age"] % dt.timedelta(days=1) == dt.timedelta(0)

	assert animals.schema["age"] == pl.Duration("us")


def test_age_follows_the_recording_window():
	"""Age is relative to the recording, so a later recording ages the same cohort."""
	later = strategies.analysis_recording(
		animal_ids=["A", "B", "C"], start="2023-06-24 00:00:00", finish="2023-06-26 23:00:00"
	)
	assert (build(later)["age"] - build()["age"]).unique().to_list() == [dt.timedelta(days=31)]
