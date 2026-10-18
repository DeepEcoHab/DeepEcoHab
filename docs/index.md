---
title: DeepEcoHab
site:
  hide_outline: true
---

DeepEcoHab analyses [EcoHab](https://elifesciences.org/articles/19532) recordings: a group of
up to 12 mice living in connected cages, tracked by RFID antennas for days. From the antenna
reads it measures where each animal spent its time, who it spent it with, who chased whom and
how the dominance ranking evolved. Every figure is interactive.

![The recording dashboard](images/readme-dashboard.png)

## Install

DeepEcoHab needs Python 3.15. [uv](https://docs.astral.sh/uv/) installs Python for you.

::::{tab-set}
:::{tab-item} Windows
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv tool install "deepecohab[app]"
deepecohab-shortcut
```
:::
:::{tab-item} Linux / macOS
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install "deepecohab[app]"
deepecohab-shortcut
```
:::
::::

Then double-click the **DeepEcoHab** icon on your desktop. The app opens in your browser.

:::{dropdown} Commands not found, older uv, updating, or using it from code
- **Commands not found:** run `uv tool update-shell` and reopen the terminal.
- **Older uv:** run `uv self update` first; an older uv only knows the 3.15 pre-releases.
- **Update:** the app offers **Update now** when a new version is out. By hand:
  `uv tool upgrade --python 3.15 "deepecohab[app]"`.
- **From code:** `uv pip install "deepecohab[notebook]"` installs the library with what
  Jupyter needs; drop `[notebook]` for plain scripts.
- **From source:** `git clone https://github.com/DeepEcoHab/DeepEcoHab.git`, then
  `pip install ".[app]"` inside it.
:::

## Where to go next

::::{grid} 1 1 2 2
:::{card} Data and pipeline
:link: data.md
Input file formats, the analysis parameters, and every table the pipeline builds.
:::
:::{card} Use the app
:link: app.md
Create a project, run the analysis, explore recordings and export figures. No code.
:::
:::{card} Analyse in Python
:link: analysis.md
The same pipeline from a script or notebook, and the tables it builds.
:::
:::{card} Plotting
:link: plotting.md
Every plot, live, with the options it takes.
:::
::::

## Team and citation

DeepEcoHab is developed at Ewelina Knapska's Laboratory of Emotions Neurobiology, Nencki
Institute of Experimental Biology, Warsaw. [Reference](reference.md) lists the contributors,
how the project is run, and how to cite it. Bugs and requests go to the
[issue tracker](https://github.com/DeepEcoHab/DeepEcoHab/issues).
