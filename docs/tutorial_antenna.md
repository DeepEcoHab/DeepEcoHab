# Step by step guide to antenna data analysis

This guide walks through analysing EcoHab antenna recordings with the DeepEcoHab Python
API: creating a project, adding recordings, running the analysis pipeline and reading its
results. Drawing figures from those results is covered in [Plotting](./plotting.md).

```python
import deepecohab as deh
```

## What a recording needs

A recording is added from two files: the antenna registrations as a parquet file, and a
metadata JSON describing the timeline, the cohort and the habitat layout.

### Antenna registrations

One row per antenna read, with exactly these columns, in this order:

| column | type | meaning |
|---|---|---|
| `datetime` | `Datetime("us", <recording timezone>)` | when the read happened |
| `antenna` | `Categorical` | antenna name (Antena-COM), as the layout names it (e.g. `"1-3"`) |
| `time_under` | `Duration("us")` | how long the transponder stayed under the antenna |
| `animal_id` | `Enum(<cohort tags, sorted>)` | the animal's RFID tag |
| `internal_board_timestamp` *(optional)* | `Datetime("us", <recording timezone>)` | the acquisition board's own clock; not used by the analysis yet |

Timestamps are microseconds in the timezone the metadata names. The schema is checked when
the recording is added, and a mismatch is rejected with a list of what differs: a missing or
unexpected column, a wrong type, or columns out of order. DeepEcoHab does not read the raw EcoHab `.txt` logs: convert them to this format
first.

### Metadata

A JSON file with a single `recording` object. Abbreviated below: `...` marks where a real
file lists more animals, cages, tunnels or antenna pairs.

```json
{
  "recording": {
    "name": "cohort1_2023_05_17",
    "project_name": "Tsc2",
    "recording_location": "006",
    "notes": "",
    "timeline": {
      "start_datetime": "2023-05-17T13:02:41+02:00",
      "end_datetime": "2023-05-22T10:02:17+02:00",
      "recording_timezone": "Europe/Warsaw",
      "phases": {"light_phase": "01:00:00", "dark_phase": "13:00:00"},
      "start_from": "dark_phase",
      "end_with": "dark_phase"
    },
    "cohort": {
      "animals": [
        {
          "tag": "1850E61A04",
          "subject_name": "ID01",
          "mouse_line": "Vglut2",
          "genotype": "HET",
          "sex": "Female",
          "date_of_birth": "2023-01-22",
          "genetic_background": "C57B",
          "treatment": "no",
          "notes": ""
        },
        ...
      ]
    },
    "layout": {
      "cages": [
        {"name": "cage_1", "cell_id": "C1", "cage_type": "social", "antennas": ["1", "8"]},
        ...
      ],
      "tunnels": [
        {
          "name": "tunnel_1",
          "tunnel_no": 1,
          "start_cell_id": "C1",
          "end_cell_id": "C2",
          "antennas": ["1", "2"]
        },
        ...
      ],
      "antenna_combinations": {"1_1": "cage_1", "1_2": "c1_c2", "2_1": "c2_c1", "2_2": "cage_2", ...},
      "tunnels_map": {"c1_c2": "tunnel_1", "c2_c1": "tunnel_1", ...}
    },
    "events": []
  }
}
```

**Timeline.** `start_datetime` and `end_datetime` carry their UTC offset;
`recording_timezone` is an IANA name. `phases` gives the onset of each phase, and
`start_from` names the phase that opens the experiment. The experiment starts at the onset
of that phase *nearest* to `start_datetime` - 13:00 in the example above - and days and
hours are counted from there. Data recorded before it is left out; if recording began after
it, the first phase is simply short. Adding a recording warns when either gap exceeds an
hour.

`end_with` (required) does the same for the end: the experiment ends where the named phase
*closes*, which is the onset of the other phase, nearest to `end_datetime`. Data recorded
after it is left out; if recording stopped before it, the last phase is simply short.

**Cohort.** Every animal field is required except `subject_name` and `treatment`, which may be
left out or set to `null`. Text fields may not be empty. `tag` is the RFID tag used in
`animal_id`. The recording's `name`, `project_name` and `recording_location` may not be empty
either.

**Layout.** `antenna_combinations` maps a pair of consecutive reads,
`"<previous antenna>_<current antenna>"`, to the position the animal was in between them:
a cage name, or a directional tunnel such as `c1_c2` (travelling from cage 1 towards cage 2).
A pair it does not list resolves to `undefined`. `tunnels_map` maps each directional tunnel to
its tunnel name.

**Events** are optional: things done during the recording, such as presenting a stimulus or
injecting the cohort. Each has a `name`, a `description` and one or more `bouts`, each bout
running from `start` up to `end`. A bout with a `position` happened in those cages or tunnels
- it takes a list, so one bout can name several - and a bout without one applies to the whole
habitat. For example, a novel object placed in cage 1 on two mornings, and an injection given
to the whole cohort:

```json
"events": [
  {
    "name": "novel_object",
    "description": "novel object placed in cage 1",
    "bouts": [
      {"start": "2023-05-18T10:00:00+02:00", "end": "2023-05-18T10:30:00+02:00", "position": ["cage_1"]},
      {"start": "2023-05-19T10:00:00+02:00", "end": "2023-05-19T11:30:00+02:00", "position": ["cage_1"]}
    ]
  },
  {
    "name": "injection",
    "description": "saline i.p., whole cohort",
    "bouts": [
      {"start": "2023-05-20T12:45:00+02:00", "end": "2023-05-20T13:15:00+02:00"}
    ]
  }
]
```

Bouts must fall inside the analysed window, bouts of one event may not overlap, and event
names must be unique. Different events may overlap each other. See [Events](#events) for the
table the analysis builds from them.

Metadata files written for older versions of DeepEcoHab may carry an `interpolate_positions`
key. It no longer exists and is rejected: delete it.

## Create a project

A project is a directory holding any number of recordings. It is created in a folder named
after the project inside `location`.

```python
project = deh.Project.create(
	project_name="tsc2",
	experimenter="Jane Doe",
	location="path/to/projects",
	description="Vglut2 HET vs WT, females",
)
```

To open an existing project instead:

```python
project = deh.Project.load("path/to/projects/tsc2")
```

The project writes a log of everything done to it. Release the file with `project.close()`,
or open the project in a `with` block:

```python
with deh.Project.load("path/to/projects/tsc2") as project:
	...
```

## Add recordings

A recording is delivered as files sharing one name, `<name>.<suffix>`:

- `<name>.config.json`, the metadata: habitat layout, light cycle, cohort and events;
- `<name>.data.parquet`, the antenna registrations;
- `<name>.diagnostic.json`, optional, the acquisition software's diagnostics.

```python
recording = project.add_recording(
	"path/to/cohort1.config.json",
	"path/to/cohort1.data.parquet",
	"path/to/cohort1.diagnostic.json",  # optional
)
```

Several at once, from their files in any order - they are grouped by name, ignoring case, so
`Cohort1.Config.json` pairs with `cohort1.data.parquet`:

```python
report = project.add_recordings(Path("path/to/recordings").iterdir())

report.added  # names of the recordings that went in
report.failed  # (name, error) for each recording or stray file that did not
report.existing  # name -> files of each recording left out because its name is taken
```

A recording that fails validation, misses a required file, or comes with two different files
of one kind (two configs, say), does not stop the others; a warning lists every failure. The
same file given twice counts once.

A name already in the project is left alone unless you pass `overwrite=True`, which replaces
that recording and discards its results - how you re-add one with an edited config, say a
different `start_from` phase:

```python
project.add_recordings(report.existing["cohort1"], overwrite=True)
```

Adding copies the data into the project, so the source files are not needed afterwards:

```
tsc2/
  project.json            which recordings the project holds
  project.log             everything done to the project
  cohort1_2023_05_17/
    config.json           the validated metadata
    raw/data.parquet      the copied registrations
    raw/config.json       the metadata exactly as it was given
    raw/diagnostic.json   the diagnostics, when they came with it
    results/              one parquet file per analysis table
```

Recordings are reached by name, or all together:

```python
recording = project["cohort1_2023_05_17"]
project.recordings
project.remove_recording("cohort1_2023_05_17")  # keeps the files
project.remove_recording("cohort1_2023_05_17", delete_files=True)  # deletes them too
```

## Run the analysis

```python
project.run_analysis()
```

This builds every analysis table for every recording, analysing recordings in parallel
under one progress bar. Tables that already exist are skipped, so running it again only
builds what is missing and an interrupted run picks up where it stopped. A recording that
fails does not stop the others; its error is raised once they have finished.

Options:

```python
project.run_analysis(
	params=deh.AnalysisParams(minimum_time=5),
	names=["cohort1_2023_05_17"],  # only these recordings
	targets=["feature_df"],  # only these tables and the tables they are built from
	overwrite=True,  # rebuild tables that already exist
	workers=2,  # how many recordings to analyse at once
)
```

Existing tables are not rebuilt when parameters change: pass `overwrite=True`, with
`targets` to limit what is rebuilt.

### Analysis parameters

| parameter | default | effect |
|---|---|---|
| `minimum_time` | `2` | seconds two animals must be together continuously for a meeting to count |
| `minimum_time_alone` | `10` | seconds an animal must be alone continuously for it to count as time alone |
| `extrapolation_limit` | `43200` | seconds an animal's last known position is carried on past its last registration; see [Activity and time alone](#activity-and-time-alone) |
| `chasing_time_window` | `(0.1, 1.2)` | shortest and longest chasing event, in seconds |
| `use_prev_ranking` | `True` | start the ranking from the previous ranking stored with the recording; see [Chasings and ranking](#chasings-and-ranking) |

## Load results

```python
activity = recording.load_results("activity_df")  # polars LazyFrame
activity = recording.load_results("activity_df", eager=True)  # polars DataFrame
```

An unknown name raises `KeyError`; a table that has not been built yet raises
`FileNotFoundError`. To list every table the pipeline builds:

```python
from deepecohab.core.data_model import DataFrameRegistry

DataFrameRegistry.list_available()
```

## The analysis tables

### Calendar columns

Most tables place their rows on the recording's calendar with four columns:

- `day` - the experiment day, from 1. A day is 24 hours from the experiment start, not a
  calendar date.
- `hour` - the hour within that day, 0 to 23. Hour 0 begins at the `start_from` onset, so it
  is not the wall-clock hour.
- `phase` - `light_phase` or `dark_phase`.
- `phase_count` - phase occurrences numbered from 1 in the order they happened, whatever
  their type.

Because days and hours count from a phase onset, the same day and hour mark the same point of
the experiment in every recording.

Not every table carries all four. `animals` and `recording_quality` describe the recording as a
whole and carry none of them. `incohort_sociability` is measured per phase occurrence, so it has
no `hour`, and `phase_durations` describes the phases themselves and carries only `phase` and
`phase_count`.

Tables described as *per hour* below hold a row for every hour, animal (or pair) and
position, with `0` where nothing happened. Positions are cage and tunnel names plus
`undefined`, which marks a pair of reads the layout does not allow, and the stretch past an
animal's last registration that the carry-forward below does not cover. `main_df`, `padded_df`,
`match_df` and `chasings_df` keep tunnels directional (`c1_c2`); every other table uses tunnel
names. Time is held as polars `Duration` columns; convert with, for example,
`pl.col("time_in_position").dt.total_seconds()`.

### Overview

| table | one row per | columns |
|---|---|---|
| `animals` | animal | the cohort metadata, plus `age` at the start of the recording |
| `main_df` | antenna read | `position`, `time_spent` since the animal's previous read, calendar columns |
| `padded_df` | minute-long piece of a visit | as `main_df`, with visits cut at every minute; `interpolated` marks all but the first piece |
| `phase_durations` | phase occurrence | `duration` |
| `event_bouts` | event bout and hour it overlaps | `event`, `position`, `start`, `end` |
| `recording_quality` | animal and antenna | `detected`, `missed`, `miss_rate` |
| `activity_df` | hour, animal and position | `time_in_position`, `visits_to_position`, `time_alone` |
| `match_df` | chasing event | `winner`, `loser`, `position`, `chasing_length` |
| `chasings_df` | hour, chaser, chased and tunnel | `chasings` |
| `ranking` | animal, after every chasing event | `mu`, `sigma`, `ordinal`, `social_rank` |
| `pairwise_meetings` | hour, pair and position | `time_together`, `pairwise_encounters` |
| `incohort_sociability` | phase occurrence and pair | `proportion_together`, `sociability` |
| `feature_df` | hour, animal, metric and position | `value`, `exposure` |

### Activity and time alone

`activity_df` holds, for every animal, position and hour:

- `time_in_position` - time spent there.
- `visits_to_position` - visits. A visit spanning several minutes is counted once, in the
  hour it began.
- `time_alone` - time spent there with no other animal present. Solitary spans no longer
  than `minimum_time_alone` are ignored, since animals travelling together arrive moments
  apart.

Each animal's last position is carried on past its final registration, so an animal that stops
moving keeps accruing time where it was last seen. Silence is only weak evidence of staying put,
though, so the carry stops `extrapolation_limit` seconds after that registration - 12 hours by
default - and the rest of the window is `undefined`. An animal taken out of the habitat
mid-recording therefore stops occupying a cage instead of sitting in one until the end.

### Chasings and ranking

A chasing event is one animal following another through a tunnel: the winner enters a tunnel
from a cage while the loser is still inside, the loser leaves between 0.1 and 1.2 seconds
(`chasing_time_window`) after the winner entered, and before the winner does. `match_df` holds
one row per event, dated by the winner's exit. `chasings_df` counts them per hour, with
`chaser` (the winner) and `chased` (the loser). It holds only the hours a pair chased in; a
missing row is zero, and `time_grid` lists every hour, for averaging over them.

`ranking` replays the chasing events in order as matches the chaser won, updating every
animal's Plackett-Luce rating after each one - the approach multiplayer games use to rank
players. Chasing a higher-ranked animal gains more than chasing a lower-ranked one. The table
holds the full trajectory: `mu` (estimated skill), `sigma` (its uncertainty), `ordinal` (a
conservative rating combining the two) and `social_rank`, which marks the animal with the
highest ordinal at that point as `dominant`, the lowest as `subordinate` and the rest as
`middle`.

To continue ranking the same animals from where an earlier recording left off:

```python
project["week_2"].set_prev_ranking(project["week_1"].load_results("ranking"))

project.run_analysis(names=["week_2"], targets=["ranking"], overwrite=True)
```

Each animal's last rating is stored as `prev_ranking.parquet` in the recording's folder, and
every later build of its ranking starts from it; `AnalysisParams(use_prev_ranking=False)`
builds one from scratch instead, and `set_prev_ranking(None)` removes it. A previous ranking
holding animals missing from the cohort is rejected. In the app, drop the earlier
`ranking.parquet`, renamed after the recording it seeds, into Parameters; with Overwrite on, a
switch picks whether the rebuild uses it.

### Pairwise meetings and in-cohort sociability

`pairwise_meetings` holds, for every pair of animals, position and hour they met in (a missing
row is zero, as in `chasings_df`):

- `time_together` - time both spent in the position at once.
- `pairwise_encounters` - meetings that began in that hour.

A meeting is one stretch of continuous co-presence; other animals arriving or leaving do not
interrupt it. Meetings no longer than `minimum_time` are dropped. Pairs are unordered, with
`animal_id` sorting before `animal_id_2`, and both cages and tunnels are covered.

`incohort_sociability` compares, per phase occurrence, how much time each pair spent together
in the cages against how much they would share by chance given each animal's own cage
occupancy {cite:p}`puscian2016ecohab`:

- `proportion_together` - time together in cages, as a fraction of the phase.
- `sociability` - that fraction minus its chance expectation, summed over cages. Positive
  values mean the pair was together more than their cage preferences alone explain.

### Feature table

`feature_df` collects per-animal metrics for comparison across animals and recordings. Each
row stores a metric's `value` and the `exposure` it arose from, both per hour:

| metric | value | exposure | rate reads as |
|---|---|---|---|
| `activity` | visits | hours observed | visits per hour |
| `time_alone` | hours alone | hours observed | fraction of time alone |
| `time_together` | hours spent with other animals | hours observed × (n_mice − 1) | fraction of time, per partner |
| `pairwise_encounters` | meetings | hours observed × (n_mice − 1) | meetings per partner-hour |
| `n_chasing` | chases as chaser | hours observed × (n_mice − 1) | chases per partner-hour |
| `n_chased` | chases as chased | hours observed × (n_mice − 1) | chases per partner-hour |
| `n_chasing_per_detection` | chases as chaser | antenna reads | chases per read |

`activity`, `time_alone`, `time_together` and `pairwise_encounters` are held per `position`,
each exposed against the time spent there, so a rate filtered to some positions reads within
them and the positions still add back up to the whole. Chasings belong to no position, so the
three chasing metrics have a `null` position.

A rate is `sum(value) / sum(exposure)` over whatever grouping you need, so the hourly rows add
up exactly to a phase, a day or the whole recording:

```python
import polars as pl

rates = (
	recording.load_results("feature_df")
	.filter(pl.col("phase") == "dark_phase")
	.group_by("animal_id", "metric")
	.agg((pl.sum("value") / pl.sum("exposure")).alias("rate"))
	.collect()
)
```

Do not average per-hour rates: hours differ in how long an animal was observed. Dividing by
partners keeps cohorts of different sizes comparable. Who wins between two animals needs no
extra metric: `n_chasing / (n_chasing + n_chased)` on the summed values.

### Recording quality

`recording_quality` shows how many passes over each antenna went unrecorded. When an animal
turns up at an antenna the layout does not connect to its previous one, the antennas on the
only shortest route between the two are counted as missed. `miss_rate` is the percentage of an
animal's passes over an antenna that were missed. It is a lower bound: many missed reads leave
no trace.

Laid out as an animal-by-antenna matrix, a faulty antenna shows up as a column and a faulty
transponder as a row:

```python
recording.load_results("recording_quality", eager=True).pivot(
	on="antenna", index="animal_id", values="miss_rate"
)
```

### Events

`event_bouts` places each bout of the recording's [events](#metadata) on the calendar, with
one row for every hour the bout overlaps. For the example events above, with the experiment
starting at the 13:00 dark onset on 17 May, it holds:

| event | position | start | end | phase | day | phase_count | hour |
|---|---|---|---|---|---|---|---|
| novel_object | cage_1 | 2023-05-18 10:00 | 2023-05-18 10:30 | light_phase | 1 | 2 | 21 |
| novel_object | cage_1 | 2023-05-19 10:00 | 2023-05-19 11:30 | light_phase | 2 | 4 | 21 |
| novel_object | cage_1 | 2023-05-19 10:00 | 2023-05-19 11:30 | light_phase | 2 | 4 | 22 |
| injection | null | 2023-05-20 12:45 | 2023-05-20 13:15 | light_phase | 3 | 6 | 23 |
| injection | null | 2023-05-20 12:45 | 2023-05-20 13:15 | dark_phase | 4 | 7 | 0 |

The second novel-object bout runs past 11:00, so it covers two hours. The injection crosses the
13:00 dark onset, which is also where day 4 begins, so its two rows fall in different days and
phases. A `null` position means the whole habitat; a bout naming
several positions gets a row per position. A bout ending exactly on the hour does not
reach into the next one.

Because the table carries the calendar columns, it joins onto any per-hour table. For example,
activity during the hours a novel object was present:

```python
bouts = recording.load_results("event_bouts")

during_novel_object = recording.load_results("activity_df").join(
	bouts.filter(pl.col("event") == "novel_object").select("day", "hour").unique(),
	on=["day", "hour"],
)
```

Plots mark event bouts on their time axes; see [Plotting](./plotting.md#common-options).

## Tube test

The tube test scores dominance from head-on tunnel encounters: two animals enter the same
tunnel from opposite ends at overlapping times, and the loser backs out to the cage it came
from.

It is not part of the pipeline: `run_analysis` never builds it, and nothing else depends on it.
Run it yourself on an analysed recording, and save the result into `results/` to load or plot
it later:

```python
from deepecohab.auxiliary_analysis.tube_test import calculate_tube_test

tube_test = calculate_tube_test(recording, max_dwell=10.0, winner_behavior="BOTH")
tube_test.sink_parquet(recording.results_path / "tube_test_df.parquet")

recording.load_results("tube_test_df")
```

- `winner_behavior` - which outcomes count: `"CHASE"` (the winner follows the loser into the
  cage it retreated to), `"GUARD"` (the winner returns to its own cage) or `"BOTH"`.
- `max_dwell` - the longest a tunnel pass may last, in seconds. Longer passes are left out,
  since they come from an animal lingering at a tunnel mouth rather than crossing.

The result counts `tube_test` events per hour, `winner`, `loser` and tunnel.

## Combine recordings

```python
table = project.generate_project_table()
```

This combines every recording's `feature_df` with its `animals` metadata, adds `recording` and
`n_mice` columns, and saves the result as `project_table.parquet` in the project directory.
A `position_type` column names what each row's position is - its cage's `cage_type`,
`"tunnel"` or `undefined` - so positions compare across recordings whose layouts name them
differently.
Pass `names=[...]` to include only some recordings, and reload a saved table with
`project.load_project_table()`.

It also adds one column per [event](#events) name declared anywhere in the project, plus
`"Any event"`, so an event can be compared against the rest of the recording without joining
`event_bouts` by hand. For each row the column reads `"During"` when a bout of that event
covered that hour, `"Same hours, other days"` when a bout covered that hour of the day on a
different day, and `"Other hours"` otherwise. A recording that does not declare the event reads
`null` throughout: it is left out of the comparison rather than counted as a control.

Rates by group come out the same way as for one recording, and an event column groups like any
other - swap in `"novel_object"` below to split a metric by whether the event was on:

```python
rates = (
	table.group_by("genotype", "phase", "metric")
	.agg((pl.sum("value") / pl.sum("exposure")).alias("rate"))
	.collect()
)
```

Recordings of different lengths contribute the days they have; filter on `day` for a common
window. Identify animals by `recording` and `animal_id` together, since transponders are reused
between cohorts.
