# Using the app

The app runs the DeepEcoHab pipeline in your browser, with no code involved. You create a project,
add your recordings, run the analysis, look through each recording, build plots across
the whole project and export figures for a paper. It all runs on your own computer: the app
reads and writes the project folders on your disk, and the browser only remembers which
projects you have opened.

This chapter follows one project from start to finish. It uses the six example recordings in
`examples/data` of the repository, so you can follow along with the same files. Each part
has a short video; the screenshots and steps underneath cover the same ground at your own pace.

## Start the app

Install DeepEcoHab with the `app` extra, as described in
[Getting started](./getting_started.md#installation). Most users just want a desktop icon to
click — create one once with:

```
deepecohab-shortcut
```

Double-click the **DeepEcoHab** icon whenever you want to work; it starts the app and opens it
in your browser. That's the whole workflow — no terminal needed again.

Prefer a terminal, or need the options below? Start it directly instead:

```
deepecohab-app
```

Your browser opens the app at `http://127.0.0.1:8050`. Leave the terminal open while you
work, and press `Ctrl+C` in it to stop the app.

| option | effect |
|---|---|
| `--port 8060` | serve on another port, if 8050 is taken |
| `--host 0.0.0.0` | accept connections from other computers on your network, not only this one |
| `--no-browser` | do not open a browser tab; go to the address yourself |
| `--debug` | print warnings and errors in the terminal instead of the log file |

Without `--debug`, warnings and errors go to `~/.deepecohab/app-cache/app.log`. That is the
file to attach when you report a problem.

When a newer version of DeepEcoHab is out, the app says so in a notice when it starts.
**Update now** closes the app, installs the new version and opens it again in a new tab;
close the old one. On Windows this happens in a new terminal window, which from then on
is the app's. Wait for any analysis to finish first: the app will not update during one.
The button is offered when DeepEcoHab was installed with `uv tool install`. Otherwise,
or to update by hand, close the app and run:

```
uv tool upgrade "deepecohab[app]"
```

```{figure} images/app/projects-empty.png
:alt: The Projects page of a fresh install, with an empty project list

The first time you start the app, the project list is empty.
```

The sidebar on the left leads to the three pages: **Projects**, **Recording dashboard** and
**Plot builder**. **Collapse**, at its foot, shrinks it to icons. The button in the top right
switches between the dark and the light theme. The app starts in the theme your system uses,
and the browser remembers your choice from then on.

```{figure} images/app/theme-light.png
:alt: The recording dashboard in the light theme

The same page in the light theme. Plots follow the theme too.
```

## Projects

A project is one folder holding any number of recordings, usually one experiment or one
line of mice. Everything starts on the Projects page.

```{figure} images/app/projects-overview.png
:alt: The Projects page with one project open and its six recordings analysed

The Projects page, with one project opened to show its recordings.
```

1. **New project** creates a project.
2. **Add project** opens a project that already exists on disk.
3. The filter shows only projects and recordings whose names contain what you type.
4. The arrow, or the project's name, opens and closes the list of its recordings.
5. How many of the project's recordings are analysed. **Project table** next to it says
   whether the table the plot builder reads has been generated, and how many rows it has.
   Once recordings are added, removed or re-analysed after the table was built, it reads
   **regenerate** instead, with a button that rebuilds it.
6. The project menu (see [The project menu](#the-project-menu)).
7. Tick recordings to analyse them.
8. Each recording's status: how many of the 13 analysis steps are done.
9. Download the recording's data, **Open** its dashboard, or remove it from the project.

For each recording, the table also shows its **Window** (first and last day, and its
time zone), how many days and phases it spans, how many mice it holds, the cohort's
traits (mouse line, genotype, sex) and how many events it declares.

<video controls preload="metadata" style="width: 100%;" title="DeepEcoHab: create a project and add recordings" src="create-project.mp4"></video>

### Create a project

1. Click **New project**.
2. Fill in **Project name** and **Experimenter**. Both are required.
3. Leave **Folder** empty to create the project in `Documents/deepecohab/projects`, or give
   another folder. Either way the project gets its own folder inside it, named after the
   project.
4. Optionally, add a **Description**: cohorts, treatment, what the project is for. It is
   shown under the project's name.
5. Click **Create project**.

```{figure} images/app/new-project-modal.png
:alt: The New project dialog with a name, an experimenter and a description filled in
:width: 60%

The New project dialog.
```

### Open an existing project

A project created earlier, in this browser or another one, or copied from a colleague,
opens with **Add project**. Paste the path of its folder, the one that holds `project.json`,
and click **Add project**.

```{figure} images/app/add-project-modal.png
:alt: The Add project dialog asking for the project folder
:width: 60%

Add project takes the folder that holds `project.json`.
```

The project list is kept by this browser only. Another browser, or another computer,
starts with an empty list; add the folders there too. Adding or removing a project from the
list never moves or deletes its files.

### Add recordings

Each recording is a set of files with the same name:

- `<name>.config.json`, the recording's metadata: its habitat layout, light cycle, cohort
  and events;
- `<name>.data.parquet`, the antenna registrations, with an `inserted` flag on the ones
  the acquisition preprocessing interpolated;
- `<name>.diagnostic.json`, the acquisition software's diagnostics, which say when
  recording stopped and restarted.

All three are required.

To add recordings:

1. Open the project's row, with the arrow or the project's name.
2. Click **Add recordings** under its table. The project menu has the same entry.
3. Drop the files onto the dialog, or click it to pick them. Add all files of every recording,
   as many recordings as you like, in one go. Names are matched ignoring case.

```{figure} images/app/add-recordings-modal.png
:alt: The Add recordings dialog with its drop zone
:width: 60%

Files group by name: `<name>.config.json`, `<name>.data.parquet` and `<name>.diagnostic.json`.
```

The files are copied into the project, so the originals stay where they were. A recording
missing a required file, one given two files of the same kind, a file that belongs to no
recording, or metadata that does not validate, is listed in the dialog under **Not added**
with the reason. Every other recording is added.

A recording whose name is already in the project asks before it goes in. **Replace**
swaps in the new files and discards the recording's results, so it must be analysed again;
this is how you re-add a recording with a corrected config. **Skip** keeps the one already
there. When several names are taken, tick the box to give the same answer for all of them.

```{figure} images/app/replace-recording-modal.png
:alt: The Replace recording dialog for a recording already in the project
:width: 60%

Replacing a recording discards its results.
```

```{figure} images/app/project-recordings.png
:alt: The project's six recordings listed as not analysed

Six recordings added. None of them is analysed yet.
```

### Run the analysis

<video controls preload="metadata" style="width: 100%;" title="DeepEcoHab: run the analysis" src="run-analysis.mp4"></video>

1. Tick the recordings to analyse. The box in the table's header ticks every recording shown.
   A bar appears at the bottom of the page.

   ```{figure} images/app/action-bar.png
   :alt: The action bar with six recordings selected

   The action bar. **Clear** unticks everything.
   ```

2. Optionally, click **Parameters** to change the analysis thresholds, then **Apply**.
   They are used by the next run you start. **Reset to defaults** puts the defaults back.

   ```{figure} images/app/params-drawer.png
   :alt: The Analysis parameters drawer
   :width: 45%

   The analysis parameters.
   ```

   | parameter | default | what it does |
   |---|---|---|
   | Minimum meeting (`minimum_time`) | 2 s | the shortest time two animals must spend together for it to count as a meeting; 0 keeps every meeting |
   | Minimum time alone (`minimum_time_alone`) | 10 s | the shortest spell an animal must spend alone for it to count; drops the brief gaps left when animals arrive moments apart |
   | Chasing window (`chasing_time_window`) | 0.1 - 1.2 s | the shortest and longest chasing event |
   | Previous ranking | none | an earlier `ranking.parquet` (or CSV) of the same animals, renamed after the selected recording it is for (`<recording>.parquet`); it is stored with that recording as soon as you drop it, and the recording's dominance ranking then starts from each animal's last rating instead of from scratch |

3. Leave **Overwrite existing tables** off to build only what is missing. Turn it on to
   rebuild every table of the selected recordings, for example after changing a
   parameter. While it is on, **Use stored previous rankings** chooses whether a rebuilt
   ranking starts from the stored previous ranking.
4. Click **Run analysis**.

Each recording counts off its steps as it runs, with the table it is building. The bar at the
bottom shows the progress of the whole run. **Cancel** stops the run once the recordings in
progress finish their current step.

```{figure} images/app/run-progress.png
:alt: Six recordings being analysed, each showing its step count

An analysis in progress.
```

When a run ends, each recording shows its status:

- **Analysed**: all 13 tables are built.
- **Partial**: some tables are built. Run the analysis again to finish it.
- **Not analysed**: no tables yet.
- **Failed**: the run stopped with an error. Hover over the badge to read it.

### The project menu

The menu at the end of a project's row (&#8942;) holds everything that acts on the project as a
whole.

```{figure} images/app/project-menu.png
:alt: The project menu
:width: 35%

The project menu.
```

- **Open in plot builder** opens the [plot builder](#plot-builder) on this project.
- **Add recordings…** is the same as the button under the recordings table.
- **Generate project table** pools the results of every analysed recording into one
  table, `project_table.parquet`, in the project folder. The plot builder reads this table.
  Generate it again after analysing new recordings; the **Project table** badge says when
  it is out of date.
- **Download** gives the project table as Parquet or CSV, or the whole project as a zip:
  every recording's config and results, with or without the raw registrations.
- **Copy path** copies the project's folder path.
- **Remove from list** takes the project off this browser's list. Its files stay on disk,
  and **Add project** brings it back.

### Download or remove a recording

The download button in a recording's row offers:

- every analysis table in one zip, as Parquet or CSV;
- the cohort as CSV;
- the recording's config as JSON;
- the raw registrations as Parquet.

```{figure} images/app/recording-download-menu.png
:alt: The download menu of one recording
:width: 35%

A recording's downloads.
```

The bin button removes a recording from the project. With two or more recordings ticked,
**Remove selected**, under the recordings table, removes them all at once. **Delist, keep
files** takes a recording off the list and leaves its folder where it is. **Reinstate**, under
the recordings table, lists the delisted recordings; tick the ones to bring back with their
results, and regenerate the project table afterwards to include them again. **Delete files**
also deletes the recording's folder: its config, raw registrations and results. Deleting
cannot be undone.

```{figure} images/app/remove-recording-modal.png
:alt: The Remove recording dialog with Delist and Delete choices
:width: 60%

Delisting keeps the files. Deleting removes them for good.
```

## Recording dashboard

**Open**, in a recording's row, takes you to the recording's dashboard. There you look through
one recording: first whether its data can be trusted, then what the animals did.

<video controls preload="metadata" style="width: 100%;" title="DeepEcoHab: the recording dashboard" src="dashboard.mp4"></video>

```{figure} images/app/dashboard-overview.png
:alt: The recording dashboard, opened on its Diagnostics tab

The recording dashboard.
```

1. The recording shown. Pick another from the list, or step through the project's
   recordings with the arrows.
2. A summary of the recording:
   - the dates it ran (hover to see the exact start and end time, and the time zone);
   - where it was recorded, the `recording_location` in its config, if it names one;
   - how many days and phases it spans, and which phase day 1 starts with;
   - the number of mice (click to jump to the cohort on Overview);
   - its cages and tunnels (click to jump to the habitat map);
   - its events (hover for their names).
3. The share of antenna passes the hardware missed, with a quality badge. Click it to
   jump to Diagnostics.
4. **Notes** on the recording.
5. **Download** the recording's data. This menu also lists every table one by one.
6. The control bar, which filters every plot on the page.
7. The tabs.
8. Each plot's own buttons (see [A plot card](#a-plot-card)).

### The control bar

```{figure} images/app/dashboard-controls.png
:alt: The control bar of the recording dashboard

The control bar.
```

- **Window** counts the recording in **Days** or in **Phases**. Each light and each dark
  phase counts as one phase.
- The first slider narrows the plots to a stretch of those days or phases.
- **Hours** narrows the plots to part of the day. Hours count from the onset of the phase
  the recording starts with, and the labels show the clock time. The coloured band under
  the slider marks the light and the dark phase.
- **Phases** switches the light and the dark phase on and off. It and **Hours** move
  together: keeping one phase narrows Hours to that phase's stretch of the day, and
  narrowing Hours ticks only the phases the window still overlaps.
- **Label** names the animals by their **Tag** or their **Name** (the `subject_name` in the
  config) on the axes, network nodes and legends, whatever colours them.
- **Animals by** colours the animals by their tag, or by any cohort attribute that differs
  between them, such as genotype or sex.
- **Group mean** draws one line per group instead of one per animal. It is available
  when the animals are coloured by a group, not by their tag.
- **Events** shows or hides the shading that marks each event on the plots with a time axis.
- **Cohort** lists the animals with their colour. Click an animal to write notes about it.

A control the recording has no use for is hidden: **Phases** when the recording holds a
single phase, **Animals by** and **Group mean** when no attribute differs between the
animals, and **Events** when the recording declares none.

```{figure} images/app/cohort-popover.png
:alt: The Cohort list, each animal with its colour, name, sex, genotype and age
:width: 40%

The cohort, as coloured on every plot.
```

A plot that cannot follow one of the controls says so with a badge, and is not redrawn when
that control changes. **Whole recording** means the plot ignores the Window. **Whole day**
means the plot has no hourly breakdown, so it ignores the Hours window. **Both phases** means
the plot does not split by phase, so it ignores the Phases filter. **Per animal** means the
plot shows one line per animal whatever Group mean says.

### The tabs

**Diagnostics** comes first, because whether the acquisition can be trusted is the question
to settle before reading anything the analysis says.

- **Detection quality**: the share of missed passes, the number of detections, the
  share of time an animal's position is unknown,
  when the recording started and ended, and how often and for how long it stopped in
  between. Each stop is drawn in red on a strip across the recording's days and phases;
  hovering one says whether it was a clean stop, an abrupt end or a silence of unknown
  cause. The bands are provisional, to be
  confirmed on more recordings: under 1% missed is good, 1-2.5% needs checking, and 2.5%
  or more is poor.

  The analysis counts whole phases from the onset of the phase day 1 starts with, so
  **Recording start** and **Recording end** show the clock time and how far each lies
  from a phase onset. A recording that started before the onset has time **lost**: what
  was recorded before the onset is left out. One that started after it has time **added**:
  the first phase has no data for that long. The end is measured the same way against
  the close of the config's `end_with` phase: a recording that ran past it has time
  **lost**, and one that stopped before it is **short**, its last phase short by that
  much. A gap of more than an hour shows as a
  warning. Hover a note for the details.
- **Habitat**: the cages, tunnels and antennas as the recording's config lays them out.
  Each antenna is tinted by how many passes it missed. Click an antenna, or focus it and
  press Enter, to see which cage end of which tunnel it sits at and its **correct**,
  **interpolated** and **bad** passes and missed share. Click elsewhere or press Escape to
  close it.
- **Reads per antenna**, every pass pooled over the cohort and stacked to 100%: read
  (**correct**), filled in by the acquisition preprocessing (**interpolated**), or still
  unresolved (**bad**), so a failing antenna stands out.
- **Missed passes by animal and antenna**, the share of each animal's passes over each
  antenna that the antenna did not read.
- **Position unknown**, the time each animal spent where the antennas could not place it,
  with its correct, interpolated and bad passes.

While recording was stopped, between two parts the diagnostics name, nothing is known:
each animal keeps its last position up to the stop and is unknown from then until its
first read after the restart. All counts cover the analysed window only.

```{figure} images/app/tab-diagnostics.png
:alt: The Diagnostics tab
```

**Overview** sums up the recording:

- **Recording at a glance**: its duration, registrations, chasings and busiest cage.
- **Feature overview**: dominance, activity and proximity measures, each z-scored, on
  one polar chart.
- **Cohort**: the cohort table. Click an animal for its notes.
- **Recording pulse**: the cohort's visits per hour, one row per day, with the hours of
  each event outlined.
- **Habitat occupancy**: the share of the cohort's time spent in each place.
- **Cohort phenotype map**: one marker per animal, showing locomotion, sociality, chasing
  and rank at once.
- **Events**, when the recording declares any: every bout on one strip across the recording,
  over its light and dark phases, then each event's bouts in a table with their day, phase,
  start, end, duration and place. Click an event's name to fold its bouts away. The card is
  drawn from the config alone, so it shows before the analysis has run.

```{figure} images/app/tab-overview.png
:alt: The Overview tab
```

**Activity** covers where the animals went and when:

- **Position timeline**: each animal's position over time, as a strip. Zoom in to see
  single visits.
- **Activity over time**: antenna detections per hour, which shows the circadian rhythm,
  or per day or phase.
- **Position preference**: how the cohort's time is spread across positions.
- **Activity per position**: visits to each position, or time spent there.
- **Position preference over time**: time in each cage or tunnel, across days or phases,
  or across the 24 hours of the day.
- **Tunnel-crossing speed**: one box per tunnel, pooling one median speed per
  animal and day or phase. Both crossing directions contribute to the same tunnel.
- **Mean tunnel-crossing speed**: speed over hours or days/phases, with a shaded
  standard-error band, side panel and event shading. The Timescale control selects
  hours or days; the shared granularity control makes the latter count phases.

Speed is the antenna-to-antenna distance divided by crossing duration, in cm/s.
The plots read per-tunnel distances from `PlotContext.tunnel_lengths_cm`.
Connecting this metadata to recording configs is pending the upstream distance
schema; this PR does not define a new config field or assume 20 cm. Until that
connection is available, speed plots require an explicitly populated context.

Both speed cards expose **Max dwell**, in seconds (default 10). Crossings must have
a positive duration no greater than this cutoff; cage visits and unresolved positions
are excluded. Shared day/phase, phase-type and hour filters apply before aggregation.
The time-course plot first averages crossing speeds within each animal/day-or-phase/hour
cell, then averages observed cells over the axis folded away. Its SEM describes variation
across those cells. Empty cells stay null and do not lower the average. Group mean
averages observed animal means and shows SEM across those animals.

```{figure} images/app/animal-speed.png
:alt: Per-tunnel boxes of daily median crossing speeds, using illustrative data

Per-tunnel speed distributions with illustrative data and configured tunnel distances.
```

```{figure} images/app/animal-speed-daily.png
:alt: Mean speed with SEM, a side panel, and event shading, using illustrative data

Mean speed over days for the same illustrative recording.
```

```{figure} images/app/tab-activity.png
:alt: The Activity tab
```

**Social** covers who spends time with whom:

- **Pairwise sociability**: how often pairs meet, or how long they spend together, per cage
  or tunnel.
- **Within-cohort sociability**: the share of time every pair spent together, or their
  sociability index: time together in the cages beyond what chance predicts.
- **Time spent alone**: time each animal spent with no other animal present.
- **Sociability network**: pairs linked by the time they spend together.
- **Relationship stability**: how stable each pair's relationship is, against the time
  they share.

```{figure} images/app/tab-social.png
:alt: The Social tab
```

**Dominance** covers chasing and rank:

- **Dominance ranking**: each animal's ranking over time, or its day-to-day stability.
- **Ranking distribution**: the probability distribution of each animal's ranking on the
  last day in the window.
- **Chasings over time**: the daily rhythm of chasing, or chasing per day or phase, counted
  for each animal as chaser or as chased.
- **Dominance network**: who chases whom, with node size showing rank.
- **Chasings matrix**: chaser against chased.

```{figure} images/app/tab-dominance.png
:alt: The Dominance tab
```

On both networks, **Hide edges below** drops the weakest links, as a percentage of the
strongest, and lays the network out again from what remains.

What each measure means, and which table it comes from, is described in the
[antenna analysis guide](./tutorial_antenna.md#the-analysis-tables).

### A plot card

```{figure} images/app/card-anatomy.png
:alt: A plot card with its title, a Whole day badge, its buttons and its footer

A plot card.
```

1. What the plot shows.
2. A badge, when the plot ignores one of the controls. This one has no hourly breakdown.
3. **Format**: titles, axis ranges and colours (see [Format a plot](#format-a-plot)).
   Formatting applies to this plot on every recording you open, for as long as the
   browser tab stays open.
4. **Export** the plot as a figure file (see [Export a plot](#export-a-plot)).
5. **Full screen**. Press `Esc` to close it.
6. The analysis tables the plot reads.

Before **Format**, the **About** button (&#9432;) explains what the plot shows and how it is
calculated. Cards that are not plots, such as the cohort, the habitat map and the events,
have a full-screen button of their own.

Some cards have options of their own under their title, such as **Mode** here. A card
whose tables are not built yet says which tables it needs. Run the analysis for the
recording to build them.

### Notes

**Notes** opens the recording's notes. Clicking an animal, in the Cohort card or in the
Cohort list, opens that animal's notes. Notes are saved with the recording, so everyone who
opens the project sees them.

```{figure} images/app/notes-modal.png
:alt: The Notes dialog
:width: 60%

Notes on a recording.
```

## Plot builder

The plot builder charts the whole project at once. It reads the project table, so
generate the table first: **Generate project table** in the
[project menu](#the-project-menu). Then open the builder from the same menu with **Open in
plot builder**, or from the sidebar: after you open a recording, the builder opens on that
recording's project. The project picker at the start of the toolbar switches to any other
project you have opened.

<video controls preload="metadata" style="width: 100%;" title="DeepEcoHab: the plot builder" src="plot-builder.mp4"></video>

```{figure} images/app/builder-overview.png
:alt: The plot builder with a box plot of time alone by genotype and sex

The plot builder, with the preset "Time alone by genotype and sex" loaded.
```

1. How **Value** is summed up (see [Value and metrics](#value-and-metrics)).
2. How many recordings, rows and metrics the project table holds.
3. **Clear shelves** takes every field off the shelves. Filters stay.
4. The plot type.
5. The fields you can plot.
6. Filters.
7. The shelves: X, Y, Colour and the rest, depending on the plot type. A shelf marked `*`
   is required.
8. The plot.
9. **Presets**: ready-made plots, and the ones you saved (see [Presets](#presets)).
10. **Save preset**. Before it is **Reset**, which goes back to the preset you last loaded;
    after it come **Format**, **Export** and **Full screen**.

### Value and metrics

The project table holds every metric in one column. Each row is one animal, in one
recording, in one hour of one day, for one metric:

- **Metric** names what the row measures;
- **Value** is how much of it there was;
- the table also records how long the animal was observed in that hour.

| metric | Value is summed up as |
|---|---|
| `activity` | visits per hour |
| `time_alone` | fraction of time |
| `time_together` | fraction of time, per partner |
| `pairwise_encounters` | encounters per partner-hour |
| `n_chasing`, `n_chased` | chasings per partner-hour |
| `n_chasing_per_detection` | chasings per detection |

Because metrics have different units, keep one metric per axis. Tick a single metric
under **Metric** in Filters, or put **Metric** on a shelf such as Facet row, so that each
metric gets its own axis. If you drop Value while several metrics are ticked, the builder
puts Metric on Facet row for you. When units would still mix, a warning above the plot
says so.

Each metric is also a field of its own, listed under **Measure** after Value and Metric and
read from that metric's rows only. Use these to put two metrics on one plot, for example
activity on X against time alone on Y.

The buttons above the presets choose how Value is summed up within each point, box or
bar:

- **Rate**, the default, is the total Value divided by the time observed, so it reads the
  same whether you pool hours, days or whole recordings;
- **Total** is the sum of Value;
- **Exposure, h** is the number of hours observed;
- **Hourly mean** is the mean over hourly rows.

### Presets

The built-in presets are a quick way to start. **Presets**, above the plot, lists them,
followed by the ones saved in the project and in this browser. Click one to see what it
shows and a picture of it, then **Load preset** and change what you like. A built-in's
picture is drawn from the example recordings; a saved preset's is the plot as it stood
when it was saved. The preset on the plot now is marked **on the plot**.

```{figure} images/app/builder-presets.png
:alt: The Presets dialog, with the list on the left and a preview of the selected preset on the right

The Presets dialog.
```

| preset | what it shows |
|---|---|
| Time alone by genotype and sex | one point per animal: its share of observed time spent without company |
| Circadian activity by genotype | visits per hour across the 24 hours after phase onset, pooled per genotype |
| Time together over the shared days | only the days every recording has, worked out from the project when the preset opens |
| Light vs dark, four metrics | four metrics side by side, each on its own axis |
| Activity around an event | activity in the hours an event ran, against the same clock hours on other days and all other hours |
| Chasing against cohort size | one point per recording: are larger cohorts more aggressive per partner? |

**Activity around an event** is listed only when the project declares events. Its preview
has an **Event** list, with the number of recordings that declare each event; pick one
before **Load preset**. **Any event** pools them all.

```{figure} images/app/builder-event-preset.png
:alt: The preview of the Activity around an event preset, with its Event list
:width: 70%

The event preset asks which event to use.
```

**Save preset** keeps the current plot, its filters and its formatting. Give it a name, and
optionally a note that the Presets dialog shows with it, and choose where to keep it:

- **In this browser**: only you see it, on this computer.
- **In the project**: it is written to `builder_presets.json` in the project folder, so
  everyone who opens the project gets it.

```{figure} images/app/preset-save-modal.png
:alt: The Save preset dialog
:width: 55%

Saving a preset.
```

A saved preset appears in the Presets dialog after the built-in ones, with a picture of the
plot as you saved it. Its bin button deletes it. Once
you change a loaded preset, the plot's title says "(edited)", and **Reset** goes back
to it.

### Build a plot

1. Pick a plot type. The shelves change with it: a box plot needs Y, a scatter plot needs
   X and Y, and a sunburst needs Path and Values.
2. Drag fields from **Fields** onto the shelves. You can also click a field and pick a
   shelf from its **Send to** menu.

   ```{figure} images/app/builder-send-menu.png
   :alt: The Send to menu of the genotype field
   :width: 30%

   Clicking a field lists the shelves that take it.
   ```

3. To remove a field, click the `×` on its chip, or drag it back to Fields.

The coloured edge of each field shows its kind:

- **measure** (orange): numbers to plot, chiefly Value;
- **category** (blue): groups, such as genotype, sex, phase or recording;
- **ordered** (green): values in order, such as day, hour, phase count, age or cohort size.

The fields come in groups:

- **Measure**: Value, Metric, and one field per metric.
- **Time**: phase, day, phase count, hour.
- **Position**: the position a row was measured at, and its **position type**: the cage's
  `cage_type` from the config, or tunnel. Position type compares positions across recordings
  whose layouts name them differently. Chasing metrics belong to no position.
- **Events**: one field per event the project declares, plus **Any event**. Each reads
  **During** in the hours the event ran, **Same hours, other days** at the same clock
  hours on the other days, and **Other hours** otherwise. A recording that does not declare
  the event is left out.
- **Animal**: tag, name and the cohort attributes.
- **Recording**: the recording's name and its number of mice.

The **Detail** shelf splits the data into groups without drawing them. Box, violin, strip,
histogram and ECDF plots fill it with recording and animal id, so each box is a spread over
animals rather than one number. The **Hover** shelf takes any number of fields of any kind
and adds them to the hover text without changing the plot. If a point covers several
values of a field, the hover text lists them, or gives how many there are when there are
more than three. Histograms and density heatmaps have no Hover shelf, because each bar or
cell counts many rows. An ordered field on a shelf can also be grouped into
blocks: click it and type, for example, `3` for blocks of three, or `1-3, 4-6`. Age, on a
shelf, can also be rounded to whole **days**, **weeks** or **months** from the same menu.

### Filters

Drop a category or ordered field on **Filters** to limit what is plotted.

```{figure} images/app/builder-filters.png
:alt: The Filters panel with metric, day and genotype filters

Filters for metric, day and genotype.
```

- A category lists its values. Tick the ones to keep; with none ticked, all are kept.
- Day, phase count, hour and number of mice offer a **Range** or **Pick values**.
- Other ordered fields, such as age, take a range.

## Format and export

The recording dashboard and the plot builder share two dialogs: **Format**, to adjust a
plot on screen, and **Export**, to render it as a file.

<video controls preload="metadata" style="width: 100%;" title="DeepEcoHab: format and export a plot" src="export.mp4"></video>

### Format a plot

```{figure} images/app/format-dialog.png
:alt: The Format dialog
:width: 40%

The Format dialog.
```

Every field left empty stays automatic, and each shows its automatic value in grey:

- **Title** (plot builder only), **X axis title** and **Y axis title**;
- the **min** and **max** of each axis;
- the colour bar's title and range, on plots that have one;
- the **Colour scale** of heatmaps and the **Category palette** of groups. A palette with
  fewer colours than the plot has groups cannot be picked;
- **Shared y axis** (plot builder only): whether facets share one y axis or each scales to
  its own data.

A field that does not apply to the plot is greyed out. A title you set stays with its axis
only while the same field is on it: put another field there and the axis goes back to
automatic. **Reset formatting** clears every field. In the builder, formatting is saved
with the preset.

### Export a plot

```{figure} images/app/export-dialog.png
:alt: The Export plot dialog with a preview on the left and the settings on the right

The Export dialog.
```

- **Style**: **Publication** renders the plot on white in a plain journal style. **App
  theme** keeps the app's look.
- **Format**: SVG, PDF or PNG. PNG adds **Resolution**, 300 or 600 dpi.
- **Size**: 85, 114 or 174 mm wide, the usual journal column widths, or **Custom**. The
  width and height are in millimetres.
- **Font size**, from 6 to 12 pt.
- **Include**: the legend, the title, the event labels, and the plotted data as CSV, which
  comes in one zip with the figure. A plot with several panels, such as a line plot and the
  totals beside it, gets one CSV per panel, numbered in drawing order.
- **File name**.

The preview on the left is the file itself, scaled down to fit. The file is laid out at the
size you chose: text is set at your font size, the legend always sits beside the plot, and
crowded tick labels are thinned out. Anything that still does not fit is listed under the
preview as a warning, such as a legend too wide for the width. Click **Download** to save
the file.

## Where things are kept

Almost everything lives in the project folder on disk:

```
my_project/
  project.json            the project
  project.log             what was done to it, and when
  project_table.parquet   what the plot builder reads
  builder_presets.json    presets saved "in the project"
  recording_name/
    config.json           metadata and notes
    raw/                  the files as uploaded: data.parquet, config.json, diagnostic.json
    results/              one Parquet file per analysis table
```

The browser keeps only two things: the list of projects you have opened, and the presets
you saved "in this browser". Clearing the browser's site data forgets both, but no project
on disk is touched. Add the projects again to bring them back.
