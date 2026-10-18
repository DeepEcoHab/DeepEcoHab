---
title: Data and pipeline
---

A **project** is a folder of **recordings**. Each recording is added from three
[input files](#input-files), then analysed: the pipeline turns its antenna reads into a set
of [tables](#analysis-tables), one Parquet file each, tuned by a few
[parameters](#analysis-parameters). The per-animal [feature table](#feature-table) of every
recording is finally pooled into the project table, which compares animals across
recordings. The app and the Python API both read and write this same folder.

(project-folder)=
## Project folder

```text
my_project/
  project.json            the project
  project.log             what was done to it, and when
  project_table.parquet   what the plot builder reads
  builder_presets.json    presets saved in the project
  recording_name/
    config.json           validated metadata and notes
    raw/                  the files as added: data.parquet, config.json, diagnostic.json
    results/              one Parquet file per analysis table
    prev_ranking.parquet  a previous ranking, when one is set
```

(input-files)=
## Input files

A recording is three files sharing one name. Names are matched ignoring case.

| file | holds |
|---|---|
| `<name>.data.parquet` | the antenna registrations |
| `<name>.config.json` | the metadata: timeline, cohort, habitat layout and events |
| `<name>.diagnostic.json` | the acquisition software's diagnostics, which say when recording stopped and restarted |

The app requires all three; from Python the diagnostics are optional. DeepEcoHab does not
read the raw EcoHab `.txt` logs; convert them to this format first.

### Registrations

One row per antenna read, with exactly these columns in this order. A mismatch is rejected
with a list of what differs.

| column | type | meaning |
|---|---|---|
| `datetime` | `Datetime("us", <recording timezone>)` | when the read happened |
| `antenna` | `Categorical` | antenna name as the layout names it, e.g. `"1"` |
| `time_under` | `Duration("us")` | how long the tag stayed under the antenna |
| `animal_id` | `Enum(<cohort tags, sorted>)` | the animal's RFID tag |
| `internal_board_timestamp` *(optional)* | `Datetime("us", <recording timezone>)` | the board's own clock; not used yet |

(metadata)=
### Metadata

:::{dropdown} Example `config.json` (abbreviated)
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
        }
      ]
    },
    "layout": {
      "cages": [
        {"name": "cage_1", "cell_id": "C1", "cage_type": "social", "antennas": ["1", "8"]}
      ],
      "tunnels": [
        {"name": "tunnel_1", "tunnel_no": 1, "start_cell_id": "C1", "end_cell_id": "C2", "antennas": ["1", "2"]}
      ],
      "antenna_combinations": {"1_1": "cage_1", "1_2": "c1_c2", "2_1": "c2_c1", "2_2": "cage_2"},
      "tunnels_map": {"c1_c2": "tunnel_1", "c2_c1": "tunnel_1"}
    },
    "events": [
      {
        "name": "novel_object",
        "description": "novel object placed in cage 1",
        "bouts": [
          {"start": "2023-05-18T10:00:00+02:00", "end": "2023-05-18T10:30:00+02:00", "position": ["cage_1"]}
        ]
      }
    ]
  }
}
```
:::

| section | rules |
|---|---|
| timeline | Datetimes carry their UTC offset; `recording_timezone` is an IANA name. `phases` gives each phase's onset. The experiment starts at the `start_from` onset nearest to `start_datetime` and ends where the `end_with` phase closes nearest to `end_datetime`. Data outside is left out; a gap of more than an hour warns. |
| cohort | Every animal field is required except `subject_name` and `treatment`. Text fields may not be empty. `tag` is the RFID tag in `animal_id`. |
| layout | `antenna_combinations` maps two consecutive reads, `"<previous>_<current>"`, to where the animal was between them: a cage, or a directional tunnel such as `c1_c2`. Unlisted pairs resolve to `undefined`. `tunnels_map` names each directional tunnel's tunnel. |
| events | Optional. Each has a unique `name`, a `description` and `bouts` from `start` to `end`. A bout's `position` lists the cages or tunnels it happened in; without one it applies to the whole habitat. Bouts must fall inside the analysed window, and bouts of one event may not overlap. |

Older configs with an `interpolate_positions` key are rejected: delete the key.

(analysis-parameters)=
## Analysis parameters

| parameter | default | effect |
|---|---|---|
| `minimum_time` | `2` s | shortest continuous time together that counts as a meeting |
| `minimum_time_alone` | `10` s | shortest continuous time alone that counts; drops gaps left by animals arriving moments apart |
| `extrapolation_limit` | `43200` s | how long an animal's last position is carried on past its last read; beyond it the position is `undefined`, so a removed animal stops occupying a cage |
| `chasing_time_window` | `(0.1, 1.2)` s | shortest and longest chasing event |
| `use_prev_ranking` | `True` | start the ranking from a stored previous ranking, if the recording has one |

(analysis-tables)=
## Analysis tables

### Calendar columns

| column | meaning |
|---|---|
| `day` | experiment day from 1; 24 hours from the experiment start, not a calendar date |
| `hour` | 0–23 within the day; hour 0 is the `start_from` onset, not midnight |
| `phase` | `light_phase` or `dark_phase` |
| `phase_count` | phase occurrences numbered from 1, whatever their type |

Because they count from a phase onset, the same day and hour mean the same point of the
experiment in every recording. Positions are cage and tunnel names plus `undefined`. Only
`main_df`, `padded_df`, `match_df` and `chasings_df` keep tunnels directional. Times are
Polars `Duration`s: `pl.col("time_in_position").dt.total_seconds()`.

### Tables

| table | one row per | columns |
|---|---|---|
| `animals` | animal | cohort metadata, plus `age` at the start |
| `main_df` | antenna read | `position`, `time_spent` since the previous read |
| `padded_df` | minute-long piece of a visit | as `main_df`; `interpolated` marks all but the first piece |
| `time_grid` | hour of the recording (two rows where a phase boundary falls inside it) | the calendar columns only |
| `phase_durations` | phase occurrence | `duration` |
| `event_bouts` | event bout × hour it overlaps | `event`, `position`, `start`, `end` |
| `recording_quality` | animal × antenna | `detected`, `missed`, `miss_rate` |
| `activity_df` | hour × animal × position | `time_in_position`, `visits_to_position`, `time_alone` |
| `match_df` | chasing event | `winner`, `loser`, `position`, `chasing_length` |
| `chasings_df` | hour × chaser × chased × tunnel | `chasings` |
| `ranking` | animal, after every chasing event | `mu`, `sigma`, `ordinal`, `social_rank` |
| `pairwise_meetings` | hour × pair × position | `time_together`, `pairwise_encounters` |
| `incohort_sociability` | phase occurrence × pair | `proportion_together`, `sociability` |
| `feature_df` | hour × animal × metric × position | `value`, `exposure` |

`chasings_df` and `pairwise_meetings` hold only the hours something happened; a missing row
is zero, and `time_grid` lists every hour, for averaging over them.

:::{dropdown} How each measure is defined
**Visits and time.** `visits_to_position` counts a visit once, in the hour it began.
`time_alone` is time in a position with no other animal there, ignoring spells shorter than
`minimum_time_alone`.

**Chasing.** One animal follows another through a tunnel: the winner enters from a cage while
the loser is still inside, and the loser leaves within `chasing_time_window` of the winner
entering, before the winner does. Events are dated by the winner's exit.

**Ranking.** Chasing events are replayed in order as matches the chaser won, updating every
animal's Plackett–Luce rating. `mu` is the estimated skill, `sigma` its uncertainty and
`ordinal` a conservative combination. `social_rank` marks the highest ordinal `dominant`, the
lowest `subordinate`, the rest `middle`.

**Meetings.** One stretch of continuous co-presence; other animals coming and going do not
break it. Meetings no longer than `minimum_time` are dropped. Pairs are unordered.

**In-cohort sociability** [@puscian2016ecohab], per phase occurrence: `proportion_together`
is the pair's time together in cages as a fraction of the phase; `sociability` subtracts what
the two animals' own cage occupancy predicts by chance. Positive means together more than
their cage preferences explain.

**Recording quality.** When an animal appears at an antenna the layout does not connect to its
previous one, the antennas on the only shortest route between them count as missed.
`miss_rate` is a lower bound: many missed reads leave no trace. Pivot it animal × antenna: a
faulty antenna shows as a column, a faulty tag as a row.
:::

(feature-table)=
### Feature table

`feature_df` stores each per-animal metric per hour as a `value` and the `exposure` it arose
from. Rates are `sum(value) / sum(exposure)` over any grouping; never average hourly rates.

| metric | value | exposure | rate reads as |
|---|---|---|---|
| `activity` | visits | hours observed | visits per hour |
| `time_alone` | hours alone | hours observed | fraction of time alone |
| `time_together` | hours with others | hours observed × (n_mice − 1) | fraction of time, per partner |
| `pairwise_encounters` | meetings | hours observed × (n_mice − 1) | meetings per partner-hour |
| `n_chasing` | chases as chaser | hours observed × (n_mice − 1) | chases per partner-hour |
| `n_chased` | chases as chased | hours observed × (n_mice − 1) | chases per partner-hour |
| `n_chasing_per_detection` | chases as chaser | antenna reads | chases per read |

Dividing by partners keeps cohorts of different sizes comparable. The first four metrics are
held per position; the chasing metrics have a `null` position.

The **project table** (`project.generate_project_table()`) stacks every recording's
`feature_df` with its cohort, plus `recording`, `n_mice`, `position_type` (the cage's
`cage_type`, `"tunnel"` or `undefined`), and one column per event plus `"Any event"`, reading
`"During"`, `"Same hours, other days"` or `"Other hours"`, or `null` for recordings that do not
declare it.
