# What is DeepEcoHab?

DeepEcoHab is a package for analysis of data acquired in the EcoHab system - a semi-naturalistic cage design for long-term recording of a group of up to 12 mice.
The package provides four modules:

## 1. Antenna analysis
`deepecohab.core` - a set of optimized, fast functions to analyze your experiments purely on information obtained from animals crossing the antennas: time spent in cages and tunnels, number of visits, time alone, chasings and dominance ranking, pairwise meetings, in-cohort sociability, a per-antenna report of how much of the movement the hardware caught, and a feature table that compares animals across recordings. Provides an approximate picture of the social structure and social hierarchy type.

## 2. Auxiliary analyses
`deepecohab.auxiliary_analysis` - analyses kept outside the routine pipeline, which you run yourself when you need them, such as the spontaneous tube test.

## 3. Plotting
`deepecohab.plotting` - interactive Plotly figures of the most important results, built straight from an analysed recording.

## 4. Web app
`deepecohab.app` - a browser GUI over the same pipeline: create a project, add recordings, run the analysis, build plots and export them without writing any code. It ships with the `app` extra and is started with `deepecohab-app`.

Analysis of pose estimation data is planned.

Organise your recordings into a project, then analyze them step by step or run the whole pipeline over every recording with one call.

## Installation

DeepEcoHab requires Python 3.12 or newer. In the spirit of open-source we recommend the [uv](https://docs.astral.sh/uv/) package and project manager to work with deepecohab.

### Step 1 — Install `uv`

**Windows:**
```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Linux / macOS:**
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Step 2 — Install DeepEcoHab

```
uv tool install "deepecohab[app]"
```

### Step 3 — Start the app

Most people never need a terminal after this. Create a desktop shortcut once:

```
deepecohab-shortcut
```

Then double-click the **DeepEcoHab** icon whenever you want to work — it starts the app and
opens it in your browser. No shortcut? Start it directly instead:

```
deepecohab-app
```

Create a project, add your recordings, run the analysis and build plots — no code involved.
This is how we expect most people to use DeepEcoHab. [Using the app](./app.md) walks through
it step by step, with screenshots and short videos.

### Updating

The app tells you when it starts if a newer version is out; **Update now** installs it and
reopens the app. To update by hand instead, close the app and run:

```
uv tool upgrade "deepecohab[app]"
```

### Working from code instead

The app is an extra because the analysis itself does not need Dash. If you only want the
library, install it with the notebook extra, which adds the Jupyter kernel and what
`figure.show()` needs inside a notebook (plain scripts can drop `[notebook]`):

```
uv pip install "deepecohab[notebook]"
```

We recommend [VSCode](https://code.visualstudio.com/download) with the Jupyter
extension to run the example notebooks provided in the repository.

To install from source:

```
cd location_to_clone_to
git clone https://github.com/DeepEcoHab/DeepEcoHab.git
cd DeepEcoHab
pip install ".[app]"
```

## Data structure of DeepEcoHab

A project is a directory. Each recording in it keeps its validated metadata, a copy of its antenna registrations, and the results of its analysis - one [Parquet](https://parquet.apache.org/) file per table, read with [Polars](https://pola.rs/):

```
my_project/
  project.json
  project.log
  recording_name/
    config.json
    raw/
      data.parquet
      config.json
    results/
      main_df.parquet
      activity_df.parquet
      ...
```

To load a table, open the project and ask the recording for it by name:

```python
import deepecohab as deh

project = deh.Project.load("path/to/my_project")
activity = project["recording_name"].load_results("activity_df")
```

Every table, and what it holds, is described in the [antenna analysis guide](./tutorial_antenna.md#the-analysis-tables).

## Data Visualization

Plots are built from an analysed recording by name:

```python
figure = deh.to_figure(deh.plot("activity-bar", project["recording_name"]))
figure.show()
```

`deh.plot` returns the plain dict Plotly draws; `deh.to_figure` turns it into a Plotly figure, which can be modified further, saved as an interactive `.html` file with `figure.write_html()`, exported as an image in a preferred format and resolution for publication with `figure.write_image()`, or saved as JSON with `figure.write_json()` and reopened with `plotly.io.read_json()`. See [Plotting](./plotting.md) for every available plot and its options.

## DeepEcoHab team

DeepEcoHab is developed at Ewelina Knapska's Laboratory of Emotions Neurobiology,
Nencki Institute of Experimental Biology, Polish Academy of Sciences, Warsaw:

- **Konrad Danielewski** ([@KonradDanielewski](https://github.com/KonradDanielewski)) - lead developer and maintainer
- **Ula Włodkowska** ([@uwlodkowska](https://github.com/uwlodkowska))
- **Marcin Lipiec** - principal investigator
- **Ewelina Knapska** - head of the laboratory

With contributions from [@Winiarsky](https://github.com/Winiarsky) and
[@Brosnan-neuro](https://github.com/Brosnan-neuro).

Found a bug or missing a feature? Open an issue at
[github.com/DeepEcoHab/DeepEcoHab/issues](https://github.com/DeepEcoHab/DeepEcoHab/issues).

## Citations

DeepEcoHab has no paper of its own yet. If you use it in published work, please cite the
package {cite}`deepecohab` together with the paper introducing the Eco-HAB system
{cite}`puscian2016ecohab`:

```bibtex
@software{deepecohab,
  title     = {{DeepEcoHab}: fast and intuitive data analysis platform for {EcoHab} experiments},
  author    = {Danielewski, Konrad and Wlodkowska, Ula and Lipiec, Marcin and Knapska, Ewelina},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/DeepEcoHab/DeepEcoHab}
}
```

```{bibliography}
```
