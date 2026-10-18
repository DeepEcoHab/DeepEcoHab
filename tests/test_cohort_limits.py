import pytest
from pydantic import ValidationError

from deepecohab.core.data_model import Cohort


def animal(tag: str) -> dict:
	return {
		"tag": tag,
		"mouse_line": "line",
		"genotype": "WT",
		"sex": "F",
		"date_of_birth": "2024-01-01",
		"genetic_background": "B6",
		"notes": "",
	}


def test_cohort_rejects_duplicate_tags():
	with pytest.raises(ValidationError, match="unique"):
		Cohort(animals=[animal("001"), animal("001")])


def test_cohort_caps_animals_at_the_bitmask_width():
	assert Cohort(animals=[animal(f"{i:03d}") for i in range(63)]).n_mice == 63
	with pytest.raises(ValidationError):
		Cohort(animals=[animal(f"{i:03d}") for i in range(64)])
