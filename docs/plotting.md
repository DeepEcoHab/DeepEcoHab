# Plotting

DeepEcoHab draws its figures straight from an analysed recording. Every plot is an
interactive [Plotly](https://plotly.com/python/) figure, so it can be zoomed, hovered,
restyled and exported. Run the analysis first, as described in the
[antenna analysis guide](./tutorial_antenna.md).

## Draw a plot

```python
import deepecohab as deh

project = deh.Project.load("path/to/projects/tsc2")
recording = project["cohort1_2023_05_17"]

figure = deh.plot("activity-bar", recording)
figure.show()
```

Options are passed as keyword arguments:

```python
deh.plot("activity-bar", recording, metric="visits", agg="mean", phase_type=["dark_phase"])
```

To make several plots of one recording, build a plot context once. It keeps the tables it
has read in memory instead of loading them again for every figure:

```python
context = deh.PlotContext.from_recording(recording)

deh.plot("chasings-heatmap", context, days_range=(2, 4))
deh.plot("ranking-line", context, mode="stability")
```

Each plot reads specific analysis tables. If one has not been built, the plot raises a
`ValueError` naming it.

An option value outside its allowed choices raises a `ValueError` listing them.

## Common options

Most plots share these options; the [plot list](#available-plots) says which each one takes.

| option | values | default | effect |
|---|---|---|---|
| `granularity` | `"day"`, `"phase_count"` | `"day"` | whether the window, and any per-unit breakdown, counts experiment days or phase occurrences |
| `days_range` | `(first, last)` | the whole recording | the window to plot, inclusive, in units of `granularity` |
| `hours_range` | `(first, last)` | every hour | the hours of the day to keep, inclusive, counted from the `start_from` onset (0 to 23) |
| `phase_type` | a list of `"light_phase"`, `"dark_phase"` | both | which phases to include |
| `agg` | `"sum"`, `"mean"` | `"sum"` | total the window, or average it; bar plots then show the spread across days or phases, and line plots a standard-error band |
| `color_by` | `"animal_id"` or a cohort attribute | `"animal_id"` | colour each animal by its tag, or by its group |
| `group_mean` | `True`, `False` | `False` | with `color_by` set to a group, draw one line, bar or marker per group instead of one per animal |
| `label_by` | `"animal_id"`, `"subject_name"` | `"animal_id"` | name each animal by its tag or its subject name on axes, network nodes and legends; every plot that shows individual animals takes it |
| `scope` | `"cages"`, `"tunnels"`, `"all"` | `"all"`, or `"cages"` for plots with one panel per position | which positions to include |
| `unit` | `"auto"`, `"seconds"`, `"minutes"`, `"hours"`, `"days"` | `"auto"` | the unit durations are shown in; `"auto"` picks the largest one the data reaches |

`days_range`, `hours_range` and `phase_type` filter by the calendar columns described in
[the analysis tables](./tutorial_antenna.md#calendar-columns): day 1 hour 0 is the onset of
the `start_from` phase.

**Colour by group.** `color_by` accepts `animal_id` and any of
`sex`, `genotype`, `treatment`, `mouse_line` and `genetic_background` that
takes more than one value in the cohort. An animal keeps its colour in every plot, whatever
the selection, and whatever `label_by` names it. To list the attributes available for a
recording:

```python
deh.available_attributes(context)
```

**Scope.** Plots drawing one panel per position (`cage-preference-evolution`,
`sociability-heatmap`) accept only `"cages"` or `"tunnels"`, since cage times run to hours
and tunnel times to seconds and cannot share one colour scale.

**Events.** Plots with a time axis (`recording-timeline`, `cage-preference-evolution`,
`activity-line`, `animal-speed-daily`, `chasings-line`, `ranking-line`, `habitat-occupancy`) shade the recording's
[event bouts](./tutorial_antenna.md#metadata), and `recording-pulse` outlines the hours they
cover.

**What a plot shows.** `deh.PlotRegistry.spec(name).info` describes what a plot shows and how
it is calculated, and `.requires` lists the tables it reads.

## Available plots

To list them in code: `deh.PlotRegistry.list_available()`.

### Recording

| plot | shows | options |
|---|---|---|
| `recording-timeline` | each animal's position over time, one row per animal, coloured by position; undefined stretches are left as gaps | `days_range`, `granularity`, `hours_range`, `label_by` |
| `recording-pulse` | the cohort's visits per hour, one row per experiment day, so gaps and rhythm drifts stand out | `days_range`, `granularity`, `hours_range` |
| `habitat-occupancy` | the share of the cohort's time held by each cage, all tunnels together and undefined, per day or phase, stacked to 100% | `days_range`, `granularity`, `hours_range` |
| `cohort-phenotype` | one marker per animal: visits against the share of its cage time spent with company, sized by dominance rating | `days_range`, `granularity`, `hours_range`, `color_by`, `label_by` |

### Recording quality

| plot | shows | options |
|---|---|---|
| `quality-antenna` | the share of passes each antenna missed, pooled over the cohort | none |
| `quality-heatmap` | the share of each animal's passes over each antenna that went unrecorded | `label_by` |

### Activity

| plot | shows | options |
|---|---|---|
| `animal-speed` | one box per tunnel, pooling one median crossing speed per animal and day or phase | `days_range`, `granularity`, `phase_type`, `hours_range`, `max_dwell` (positive seconds, default `10`) |
| `animal-speed-daily` | mean crossing speed with SEM and a side panel, across hours or days/phases; missing bins stay null | `timescale` (`"days"`, `"hours"`), `days_range`, `granularity`, `phase_type`, `hours_range`, `max_dwell`, `color_by`, `label_by`, `group_mean` |
| `activity-bar` | visits to each position, or time spent there, per animal | `metric` (`"time"`, `"visits"`), `days_range`, `granularity`, `hours_range`, `phase_type`, `agg`, `scope`, `color_by`, `group_mean`, `label_by`, `unit` |
| `time-alone-bar` | time each animal spent with no other animal present, per position | `days_range`, `granularity`, `hours_range`, `phase_type`, `agg`, `scope`, `color_by`, `group_mean`, `label_by`, `unit` |
| `cage-preference` | how the cohort's time is distributed across positions | `days_range`, `granularity`, `hours_range`, `phase_type`, `scope`, `unit` |
| `cage-preference-evolution` | time per animal in each cage or tunnel, across days or phases, or with `timescale="hours"` across the 24 hours of the experiment day | `timescale` (`"days"`, `"hours"`), `days_range`, `granularity`, `hours_range`, `agg`, `scope`, `label_by`, `unit` |
| `activity-line` | antenna reads per hour of the day: the circadian rhythm, or with `timescale="days"` per day or phase | `timescale` (`"hours"`, `"days"`), `days_range`, `granularity`, `hours_range`, `agg`, `color_by`, `group_mean`, `label_by` |

`activity-bar` with `scope="all"` keeps the undefined position, so the time no antenna could
place stays visible.

The speed plots require explicit per-tunnel distances in `PlotContext.tunnel_lengths_cm`.
Reading those distances from recording configs is pending the upstream schema; no default
distance is assumed.

### Social hierarchy

| plot | shows | options |
|---|---|---|
| `chasings-line` | chasings per hour of the day, or with `timescale="days"` per day or phase, counted for each animal as chaser or, with `scope="chased"`, as chased | `timescale` (`"hours"`, `"days"`), `scope` (`"chaser"`, `"chased"`), `days_range`, `granularity`, `hours_range`, `agg`, `color_by`, `group_mean`, `label_by` |
| `chasings-heatmap` | chaser-versus-chased matrix; columns are chasers, rows are chased | `days_range`, `granularity`, `hours_range`, `phase_type`, `agg`, `label_by` |
| `ranking-line` | each animal's ranking (`ordinal`) over time, or with `mode="stability"` its rank order in each day or phase | `mode` (`"intime"`, `"stability"`), `days_range`, `granularity`, `color_by`, `label_by` |
| `ranking-distribution-line` | the probability density of each animal's rating on the last day or phase of the window | `days_range`, `granularity`, `color_by`, `label_by` |
| `network-dominance` | directed network of chasings, with node size showing ranking | `layout` (`"spring"`, `"circular"`), `edge_cutoff`, `days_range`, `granularity`, `color_by`, `label_by` |
| `tube-test-heatmap` | winner-versus-loser matrix of tube-test outcomes; needs the [tube test](./tutorial_antenna.md#tube-test) saved first | `days_range`, `granularity`, `hours_range`, `phase_type`, `agg`, `label_by` |

### Sociability

| plot | shows | options |
|---|---|---|
| `sociability-heatmap` | time together, or number of meetings, for every pair, one panel per cage or tunnel | `metric` (`"time_together"`, `"pairwise_encounters"`), `days_range`, `granularity`, `hours_range`, `phase_type`, `agg`, `scope`, `label_by`, `unit` |
| `cohort-heatmap` | the proportion of time together, or in-cohort sociability, for every pair | `metric` (`"proportion_together"`, `"sociability"`), `days_range`, `granularity`, `phase_type`, `scope`, `label_by` |
| `social-stability` | each pair's typical proportion of time together against how steady it stays across days or phases | `days_range`, `granularity`, `phase_type`, `scope`, `color_by`, `label_by` |
| `network-sociability` | undirected network weighted by the proportion of time each pair spends together | `layout` (`"spring"`, `"circular"`), `edge_cutoff`, `days_range`, `granularity`, `scope`, `color_by`, `label_by` |

In `cohort-heatmap`, `scope` changes `proportion_together` only: `sociability` is always
measured in cages, against chance cage occupancy.

In both networks, `edge_cutoff` drops edges weaker than that percentage of the strongest before
the layout is fitted, so the network is laid out from what remains.

### Overview

| plot | shows | options |
|---|---|---|
| `metrics-polar-line` | every metric of the feature table on one polar chart, as z-scored rates | `days_range`, `granularity`, `hours_range`, `phase_type`, `color_by`, `group_mean`, `label_by` |

The z-scores are computed for display across the animals of the recording, so they compare
animals within it, not across recordings. Compare recordings with the rates in the
[project table](./tutorial_antenna.md#combine-recordings).

## Themes

Three themes are included: `light` and `dark`, as in the app, and `publication`, a plain
journal style on white. To use one for every figure in the session:

```python
import plotly.io as pio

pio.templates.default = "light"  # or "dark", "publication"
```

Without this, figures use Plotly's default. To theme a single figure:
`figure.update_layout(template="dark")`.

## Save a figure

```python
figure.write_html("activity.html")  # interactive, opens in any browser
figure.write_image("activity.png", scale=3)  # static image for publication
figure.write_json("activity.json")  # reopen later with plotly.io.read_json()
```

To lay a figure out for a journal page, as the app's Export dialog does, give its physical
size and font size. Fonts scale from `pt`, the legend moves beside the plot and crowded tick
labels are thinned:

```python
figure.update_layout(template="publication")

warnings = deh.export_figure(
	figure, "activity.pdf", width_mm=85, height_mm=60, pt=8, format="pdf"
)
```

`format` is `"svg"`, `"pdf"` or `"png"`; a PNG also takes `dpi` (300 by default). `title`,
`show_legend` and `show_events` choose what is drawn. The returned list names anything that
still does not fit, such as a legend too wide for the width. `deh.fit_for_export` does the
layout alone and returns the fitted figure with those warnings, without writing a file.
