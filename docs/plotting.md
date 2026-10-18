---
title: Plotting
site:
  hide_outline: true
kernelspec:
  name: python3
  display_name: Python 3
---

Every plot is an interactive [Plotly](https://plotly.com/python/) figure: hover, zoom, and
click legend entries to hide them. The figures below are drawn live from one example
recording each time the docs are built, grouped as the recording dashboard's tabs group
them. From Python, draw one by name:

```{code-cell} python
:tags: [remove-cell]
import functools, tempfile, warnings
import tqdm.auto
from pathlib import Path

# the progress bar is a notebook widget a static page cannot draw
tqdm.auto.tqdm = functools.partial(tqdm.auto.tqdm, disable=True)
warnings.simplefilter("ignore")
import deepecohab as deh
import plotly.io as pio
from deepecohab.plotting import theme

project = deh.Project.create(project_name="example", experimenter="docs", location=tempfile.mkdtemp())
project.add_recordings(Path("../examples/data").iterdir())
project.run_analysis(names=["Marcin_2023_05_17_Vglut2_WT_F"])
recording = project["Marcin_2023_05_17_Vglut2_WT_F"]

# As the dashboard draws its cards: no mode bar, the app's theme and card height, and the
# looks plots.mjs repaints a figure with when the site switches between light and dark.
pio.renderers.default = "plotly_mimetype"
pio.renderers["plotly_mimetype"].config = {"displayModeBar": False, "responsive": True}
looks = {name: theme.looks(name) for name in ("light", "dark")}


def card(name, height, **options):
    figure = theme.apply(deh.plot(name, recording, **options), "light")
    figure["layout"]["height"] = height
    figure["layout"]["meta"] = {**figure["layout"].get("meta", {}), "dehLooks": looks}
    return deh.to_figure(figure)
```

:::{anywidget} plots.mjs
{}
:::

```{code-cell} python
:tags: [remove-output]
figure = deh.to_figure(deh.plot("activity-line", recording, agg="mean"))
figure
```

```{code-cell} python
:tags: [remove-cell]
:label: fig-example
card("activity-line", 360, agg="mean")
```

:::::{div}
:class: col-page-right
![](#fig-example)
:::::

`deh.plot` returns the plain dict Plotly draws; `deh.to_figure` turns it into a Plotly
figure you can modify, `show()` or save. Options are keyword arguments. For several plots of
one recording, build a context once so tables are read only once:

```python
context = deh.PlotContext.from_recording(recording)
deh.plot("chasings-heatmap", context, days_range=(2, 4))
deh.plot("ranking-line", context, mode="stability")
```

`deh.PlotRegistry.list_available()` lists every plot, and `deh.PlotRegistry.spec(name).info`
explains one.

## Gallery

The recording dashboard's tabs, each card under its plot's name.

```{code-cell} python
:tags: [remove-cell]
:label: fig-quality-antenna
card('quality-antenna', 500)
```

```{code-cell} python
:tags: [remove-cell]
:label: fig-quality-heatmap
card('quality-heatmap', 430)
```

```{code-cell} python
:tags: [remove-cell]
:label: fig-metrics-polar-line
card('metrics-polar-line', 500)
```

```{code-cell} python
:tags: [remove-cell]
:label: fig-recording-pulse
card('recording-pulse', 380)
```

```{code-cell} python
:tags: [remove-cell]
:label: fig-habitat-occupancy
card('habitat-occupancy', 360)
```

```{code-cell} python
:tags: [remove-cell]
:label: fig-cohort-phenotype
card('cohort-phenotype', 360)
```

```{code-cell} python
:tags: [remove-cell]
:label: fig-recording-timeline
card('recording-timeline', 340, days_range=(1, 2))
```

```{code-cell} python
:tags: [remove-cell]
:label: fig-activity-line
card('activity-line', 360)
```

```{code-cell} python
:tags: [remove-cell]
:label: fig-cage-preference
card('cage-preference', 360)
```

```{code-cell} python
:tags: [remove-cell]
:label: fig-activity-bar
card('activity-bar', 320)
```

```{code-cell} python
:tags: [remove-cell]
:label: fig-cage-preference-evolution
card('cage-preference-evolution', 660)
```

```{code-cell} python
:tags: [remove-cell]
:label: fig-sociability-heatmap
card('sociability-heatmap', 340)
```

```{code-cell} python
:tags: [remove-cell]
:label: fig-cohort-heatmap
card('cohort-heatmap', 380)
```

```{code-cell} python
:tags: [remove-cell]
:label: fig-time-alone-bar
card('time-alone-bar', 380)
```

```{code-cell} python
:tags: [remove-cell]
:label: fig-network-sociability
card('network-sociability', 424)
```

```{code-cell} python
:tags: [remove-cell]
:label: fig-social-stability
card('social-stability', 424)
```

```{code-cell} python
:tags: [remove-cell]
:label: fig-ranking-line
card('ranking-line', 360)
```

```{code-cell} python
:tags: [remove-cell]
:label: fig-chasings-line
card('chasings-line', 360)
```

```{code-cell} python
:tags: [remove-cell]
:label: fig-ranking-distribution-line
card('ranking-distribution-line', 360)
```

```{code-cell} python
:tags: [remove-cell]
:label: fig-network-dominance
card('network-dominance', 424)
```

```{code-cell} python
:tags: [remove-cell]
:label: fig-chasings-heatmap
card('chasings-heatmap', 424)
```

### Diagnostics

:::::{div}
:class: col-page-right
::::{tab-set}
:::{tab-item} quality-antenna
The share of passes each antenna missed, pooled over the cohort. A failing antenna stands out.

Options: none.

![](#fig-quality-antenna)
:::
:::{tab-item} quality-heatmap
The share of each animal's passes over each antenna that went unrecorded. A faulty antenna is a column, a faulty tag a row.

Options: `label_by`.

![](#fig-quality-heatmap)
:::
::::
:::::

### Overview

:::::{div}
:class: col-page-right
::::{tab-set}
:::{tab-item} metrics-polar-line
Every feature-table metric on one polar chart, z-scored across the animals of this recording.

Options: `days_range`, `granularity`, `hours_range`, `phase_type`, `color_by`, `group_mean`, `label_by`.

![](#fig-metrics-polar-line)
:::
:::{tab-item} recording-pulse
The cohort's visits per hour, one row per day, so gaps and rhythm drifts stand out. Event hours are outlined.

Options: `days_range`, `granularity`, `hours_range`.

![](#fig-recording-pulse)
:::
:::{tab-item} habitat-occupancy
The share of the cohort's time in each cage, the tunnels and undefined, per day or phase.

Options: `days_range`, `granularity`, `hours_range`.

![](#fig-habitat-occupancy)
:::
:::{tab-item} cohort-phenotype
One marker per animal: visits against the share of cage time spent with company, sized by dominance.

Options: `days_range`, `granularity`, `hours_range`, `color_by`, `label_by`.

![](#fig-cohort-phenotype)
:::
::::
:::::

### Activity

:::::{div}
:class: col-page-right
::::{tab-set}
:::{tab-item} recording-timeline
Each animal's position over time, one row per animal. Zoom in to see single visits.

Options: `days_range`, `granularity`, `hours_range`, `label_by`.

![](#fig-recording-timeline)
:::
:::{tab-item} activity-line
Antenna reads per hour of the day (the circadian rhythm), or per day with `timescale="days"`.

Options: `timescale`, `days_range`, `granularity`, `hours_range`, `agg`, `color_by`, `group_mean`, `label_by`.

![](#fig-activity-line)
:::
:::{tab-item} cage-preference
How the cohort's time is spread across positions.

Options: `days_range`, `granularity`, `hours_range`, `phase_type`, `scope`, `unit`.

![](#fig-cage-preference)
:::
:::{tab-item} activity-bar
Visits to each position, or time spent there, per animal.

Options: `metric` (`time`, `visits`), `days_range`, `granularity`, `hours_range`, `phase_type`, `agg`, `scope`, `color_by`, `group_mean`, `label_by`, `unit`.

![](#fig-activity-bar)
:::
:::{tab-item} cage-preference-evolution
Time per animal in each cage, across days, or across the 24 hours with `timescale="hours"`.

Options: `timescale`, `days_range`, `granularity`, `hours_range`, `agg`, `scope`, `label_by`, `unit`.

![](#fig-cage-preference-evolution)
:::
::::
:::::

### Social

:::::{div}
:class: col-page-right
::::{tab-set}
:::{tab-item} sociability-heatmap
Time together, or number of meetings, for every pair, one panel per cage.

Options: `metric` (`time_together`, `pairwise_encounters`), `days_range`, `granularity`, `hours_range`, `phase_type`, `agg`, `scope`, `label_by`, `unit`.

![](#fig-sociability-heatmap)
:::
:::{tab-item} cohort-heatmap
The proportion of time together, or in-cohort sociability (time together beyond chance), for every pair.

Options: `metric` (`proportion_together`, `sociability`), `days_range`, `granularity`, `phase_type`, `scope`, `label_by`.

![](#fig-cohort-heatmap)
:::
:::{tab-item} time-alone-bar
Time each animal spent with no other animal present, per position.

Options: `days_range`, `granularity`, `hours_range`, `phase_type`, `agg`, `scope`, `color_by`, `group_mean`, `label_by`, `unit`.

![](#fig-time-alone-bar)
:::
:::{tab-item} network-sociability
Pairs linked by the proportion of time they spend together.

Options: `layout`, `edge_cutoff`, `days_range`, `granularity`, `scope`, `color_by`, `label_by`.

![](#fig-network-sociability)
:::
:::{tab-item} social-stability
Each pair's typical proportion of time together against how steady it stays across days.

Options: `days_range`, `granularity`, `phase_type`, `scope`, `color_by`, `label_by`.

![](#fig-social-stability)
:::
::::
:::::

### Dominance

:::::{div}
:class: col-page-right
::::{tab-set}
:::{tab-item} ranking-line
Each animal's rating over time, or its rank order per day with `mode="stability"`.

Options: `mode`, `days_range`, `granularity`, `color_by`, `label_by`.

![](#fig-ranking-line)
:::
:::{tab-item} chasings-line
Chasings per hour of the day, or per day, counted as chaser or, with `scope="chased"`, as chased.

Options: `timescale`, `scope`, `days_range`, `granularity`, `hours_range`, `agg`, `color_by`, `group_mean`, `label_by`.

![](#fig-chasings-line)
:::
:::{tab-item} ranking-distribution-line
The probability density of each animal's rating on the last day of the window.

Options: `days_range`, `granularity`, `color_by`, `label_by`.

![](#fig-ranking-distribution-line)
:::
:::{tab-item} network-dominance
Who chases whom; node size shows rank.

Options: `layout` (`spring`, `circular`), `edge_cutoff`, `days_range`, `granularity`, `color_by`, `label_by`.

![](#fig-network-dominance)
:::
:::{tab-item} chasings-heatmap
Chaser (columns) against chased (rows).

Options: `days_range`, `granularity`, `hours_range`, `phase_type`, `agg`, `label_by`.

![](#fig-chasings-heatmap)
:::
::::
:::::

The `tube-test-heatmap` plot draws winners against losers of the
[tube test](analysis.md#tube-test), once it has been saved.

## Common options

| option | values | default | effect |
|---|---|---|---|
| `granularity` | `"day"`, `"phase_count"` | `"day"` | count the window in days or in phases |
| `days_range` | `(first, last)` | whole recording | the window, inclusive, in units of `granularity` |
| `hours_range` | `(first, last)` | every hour | hours of the day to keep, 0–23 from the `start_from` onset |
| `phase_type` | `["light_phase"]`, `["dark_phase"]` | both | which phases to include |
| `agg` | `"sum"`, `"mean"` | `"sum"` | total the window, or average it with a spread across days |
| `color_by` | `"animal_id"` or a cohort attribute | `"animal_id"` | colour animals by tag, or by a group such as `genotype` |
| `group_mean` | `True`, `False` | `False` | one line or bar per group instead of per animal |
| `label_by` | `"animal_id"`, `"subject_name"` | `"animal_id"` | name animals by tag or by name |
| `scope` | `"cages"`, `"tunnels"`, `"all"` | depends on the plot | which positions to include |
| `unit` | `"auto"`, `"seconds"` … `"days"` | `"auto"` | the unit durations are shown in |

An option outside its allowed values raises a `ValueError` listing them; so does a plot whose
tables have not been built. `deh.available_attributes(context)` lists the cohort attributes
`color_by` accepts. An animal keeps its colour on every plot.

Plots with a time axis mark the recording's [events](data.md#metadata): one lane per
event above the plot, and a grey wash over the hours it ran.

## Themes and saving

Three themes are included: `light` and `dark`, as in the app, and `publication`, a plain
journal style. They are registered when `deepecohab.plotting` is imported:

```python
import deepecohab.plotting
import plotly.io as pio

pio.templates.default = "light"  # every figure in the session
figure.update_layout(template="dark")  # one figure
```

```python
figure.write_html("activity.html")  # interactive, opens in any browser
figure.write_json("activity.json")  # reopen with plotly.io.read_json()
```

To lay a figure out for a journal page, as the app's Export dialog does:

```python
figure.update_layout(template="publication")
warnings = deh.export_figure(figure, "activity.pdf", width_mm=85, height_mm=60, pt=8, format="pdf")
```

`format` is `"svg"`, `"pdf"` or `"png"` (with `dpi`, 300 by default); `title`, `show_legend`
and `show_events` choose what is drawn. The returned list names anything that does not fit at
that size. `deh.fit_for_export` returns the fitted figure without writing a file.
