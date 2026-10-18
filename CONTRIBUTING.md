# Contributing to DeepEcoHab

## Development setup

We use [uv](https://docs.astral.sh/uv/) for environment and dependency management.
The `dev` extra pulls in everything needed to work on the project — the test suite
plus the lint/format/type tooling (`ruff`, `ty`, `pre-commit`):

```bash
uv sync --extra dev
uv run pre-commit install   # enable the git hooks locally
```

## Running checks

The same hooks run locally and in CI (ruff check, ruff format, ty):

```bash
uv run --locked --extra dev pre-commit run --all-files
```

Test suite (the slow end-to-end pipeline is marked `e2e`):

```bash
uv run --locked --extra test pytest -m "not e2e"   # fast suite
uv run --locked --extra test pytest -m e2e         # full pipeline on example data
```

## TODO / FIXME / XXX / HACK / BUG comments

Every such comment names its owner (GitHub handle) and the issue that tracks it, on the
same line. Open an issue first if none exists; upstream issues (`owner/repo#N`) count too:

```python
# TODO(KonradDanielewski): drop the legacy loader #42
# FIXME(ula-w): off by one on DST days https://github.com/DeepEcoHab/DeepEcoHab/issues/57
```

```js
// HACK(KonradDanielewski): delete once Dash ships plotly/dash#3916
```

`NOTE` is not a tracked tag: write the explanation as a plain comment. The
`todo-owner-issue` pre-commit hook enforces this in Python, JS and CSS, locally and in CI.

## Updating dev tooling

There is intentionally no scheduled bot for this — refresh the tooling by hand
whenever you like, then open a normal PR so it runs through CI. Two channels are
needed because `pre-commit autoupdate` only bumps *remote* hook revisions, while
`ty` is a `repo: local` hook resolved from the lockfile:

```bash
# 1. bump remote pre-commit hook revs (ruff-pre-commit, etc.)
uv run --extra dev pre-commit autoupdate

# 2. refresh the locked dev tooling (ty is uv-managed; ruff kept in sync with its hook rev)
uv lock --upgrade-package ruff --upgrade-package ty --upgrade-package pre-commit

# 3. verify everything still passes before committing
uv run --locked --extra dev pre-commit run --all-files
```

Commit the resulting `.pre-commit-config.yaml` and `uv.lock` changes together.
