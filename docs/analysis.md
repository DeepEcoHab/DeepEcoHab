---
title: Analyse in Python
kernelspec:
  name: python3
  display_name: Python 3
---

The app is a front end to the `deepecohab` library. Everything it does is available from a
script or notebook. The code on this page runs on the example recordings each time the docs
are built, so the outputs are real.

```{code-cell} python
:tags: [remove-cell]
import functools, os, shutil, tempfile, warnings
import tqdm.auto
from pathlib import Path

# the progress bar is a notebook widget a static page cannot draw
tqdm.auto.tqdm = functools.partial(tqdm.auto.tqdm, disable=True)
warnings.simplefilter("ignore")
examples = Path("../examples/data").resolve()
os.chdir(tempfile.mkdtemp())
shutil.copytree(examples, "examples/data")
```

## Create a project and add recordings

A project is a folder holding any number of recordings. Each recording is three files sharing
a name, `<name>.config.json`, `<name>.data.parquet` and `<name>.diagnostic.json`; their format
is in [Input files](data.md#input-files). Files are grouped by name, so a whole folder
can be added at once.

```{code-cell} python
import deepecohab as deh
from pathlib import Path

project = deh.Project.create(
    project_name="vglut2",
    experimenter="Jane Doe",
    location="projects",
    description="Vglut2 HET vs WT",
)
report = project.add_recordings(Path("examples/data").iterdir())
report.added
```

`report.failed` lists what did not go in, and why. A name already in the project is skipped
unless you pass `overwrite=True`, which replaces the recording and discards its results. Open
an existing project with `deh.Project.load("projects/vglut2")`.

## Run the analysis

```{code-cell} python
project.run_analysis()
```

This builds every table for every recording, several recordings at once. Tables that exist
are skipped, so an interrupted run picks up where it stopped. To change a parameter, rebuild:

```python
project.run_analysis(
	params=deh.AnalysisParams(minimum_time=5),
	names=["Marcin_2023_05_17_Vglut2_WT_F"],  # only these recordings
	targets=["pairwise_meetings"],  # only these tables and what they need
	overwrite=True,
)
```

The parameters are listed in [Analysis parameters](data.md#analysis-parameters).

## Load results

Each table is a [Polars](https://pola.rs/) frame, loaded by name. Every table and its columns
are listed in [Analysis tables](data.md#analysis-tables).

```{code-cell} python
recording = project["Marcin_2023_05_17_Vglut2_WT_F"]
recording.load_results("activity_df", eager=True).head()
```

`load_results` returns a lazy frame unless you pass `eager=True`.

## Rates per animal

`feature_df` holds every per-animal metric as a `value` and the `exposure` it arose from,
per hour. A rate is `sum(value) / sum(exposure)` over any grouping, so hours add up exactly
to phases, days or the whole recording. Never average hourly rates.

```{code-cell} python
import polars as pl

(
    recording.load_results("feature_df")
    .filter(pl.col("phase") == "dark_phase")
    .group_by("animal_id", "metric")
    .agg(rate=pl.sum("value") / pl.sum("exposure"))
    .collect()
    .pivot(on="metric", index="animal_id", values="rate")
    .sort("animal_id")
)
```

## Compare recordings

The project table pools every recording's `feature_df` with its cohort metadata, so groups
compare the same way:

```{code-cell} python
table = project.generate_project_table()

(
    table.filter(pl.col("metric") == "time_alone")
    .group_by("genotype", "sex", "phase")
    .agg(time_alone=pl.sum("value") / pl.sum("exposure"))
    .collect()
    .sort("genotype", "sex", "phase")
)
```

It is saved as `project_table.parquet`; reload it with `project.load_project_table()`.
Identify animals by `recording` and `animal_id` together, since tags are reused between
cohorts. Recordings of different lengths contribute the days they have; filter on `day` for a
common window.

## Continue a ranking

To continue ranking the same animals from where an earlier recording left off:

```python
project["week_2"].set_prev_ranking(project["week_1"].load_results("ranking"))
project.run_analysis(names=["week_2"], targets=["ranking"], overwrite=True)
```

The previous ratings are stored with `week_2` and used by every later build of its ranking.
`AnalysisParams(use_prev_ranking=False)` ignores them, and `set_prev_ranking(None)` removes
them.

## Tube test

The tube test scores dominance from head-on tunnel encounters, where the loser backs out. It
is not part of the pipeline; run it on an analysed recording:

```python
from deepecohab.auxiliary_analysis.tube_test import calculate_tube_test

tube_test = calculate_tube_test(recording, max_dwell=10.0, winner_behavior="BOTH")
tube_test.sink_parquet(recording.results_path / "tube_test_df.parquet")
```

`winner_behavior` counts outcomes where the winner follows the loser (`"CHASE"`), returns to
its own cage (`"GUARD"`), or `"BOTH"`. Passes longer than `max_dwell` seconds are left out.
Once saved, the `tube-test-heatmap` plot can draw it.
