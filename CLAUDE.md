# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A pipeline that predicts each NBA player's *next-season* fantasy points and turns those
predictions into a fantasy-draft board. Data is pulled from the NBA API and Basketball
Reference, engineered into player-season features, and fed to an XGBoost regressor.
The end product is a ranked, position-grouped draft board with upside/downside ranges.

The orchestration lives in `run_model.ipynb`; all reusable logic lives in the `.py`
modules and is imported into the notebook. New logic should generally be written as
functions in the appropriate module, not left inline in the notebook.

## Environment & running

Dependencies are managed with **uv** (`pyproject.toml` + `uv.lock`); Python is pinned to
3.14 via `.python-version`. There is no `requirements.txt`.

```bash
uv sync                       # create/refresh .venv from the lockfile (runtime deps only)
uv sync --group dev           # also install Jupyter tooling (jupyterlab, ipykernel)
uv run jupyter lab run_model.ipynb   # primary entry point — run cells top to bottom
uv run python -c "import modeling"   # run any script inside the managed env
uv add <pkg> / uv remove <pkg>       # change deps (updates pyproject.toml + uv.lock)
```

There is no test suite, linter config, or build step. The notebook *is* the runner.
`[tool.uv] package = false` — this is a flat-module project, not an installable package.

**Version gotcha:** `hyperopt` imports `pkg_resources`, which was removed in
`setuptools>=81`. `setuptools` is therefore pinned `<81` in `pyproject.toml`; don't
loosen that bound or `import hyperopt` (and thus `config.py`) will fail on Python 3.14.

## Data flow (the big picture)

The pipeline is a linear sequence, each stage consuming the previous stage's DataFrame:

1. **Acquisition** — `data_io.load_or_generate_data(START_YEAR, END_YEAR)` returns
   `(api_df, bref_df)`. It reads cached CSVs from `data/` if present, otherwise scrapes
   the NBA API (`get_multi_season_base_and_advanced_stats`) and Basketball Reference
   (`get_combined_bref_data`) and writes the CSVs. **Scraping is slow and rate-limited**
   (`time.sleep` between calls); never delete the cached CSVs in `data/` without reason.
2. **Cleaning** — `data_cleaning.clean_and_refine_nba_data` normalizes column names to
   snake_case and coerces numerics. Columns are then subset to `SELECTED_COLUMNS` (API)
   and `BREF_COLS` (BRef).
3. **Merging** — `data_cleaning.merge_nba_dataframes` left-joins the two sources on
   `["player_name_clean", "year"]` into `combined_df`.
4. **Target + features** — `feature_engineering.calculate_fantasy_points` creates the
   `fantasy_points` target, then `feature_engineering.run_feature_pipeline` chains every
   feature step and produces the modeling matrix.
5. **Modeling** — `modeling.split_data_nba` → `tune_xgb_nba` → `create_model_nba`, with
   `create_baseline_nba` as a reference point and `generate_prediction_intervals` for
   bootstrapped ranges.
6. **Prediction + draft board** — predictions for `PRED_YEAR` are blended with scraped
   FantasyPros projections, assigned position groups, and ranked into a draft board.
   These later stages currently live as inline functions in the notebook (see TODOs).

## The two join keys that hold everything together

Nearly every merge depends on `player_name_clean` and `year`. Understand both before
touching any join:

- **`player_name_clean`** — produced by `data_cleaning.clean_name`: strips accents,
  punctuation, and suffixes (`jr`, `iii`), lowercases. Because names are the join key
  across three data sources, mismatches silently drop players. `config.NAME_OVERRIDES`
  patches known collisions; add to it when a player fails to join rather than editing
  `clean_name`. FantasyPros joins are a special case — it inconsistently includes
  suffixes, so the notebook keeps a `HOLDOUT_SUFFIX_NAMES` set.
- **`year`** — always the **season END year** (2022-23 season → `2023`), unified across
  the NBA API, BRef URLs, and draft data. Preserve this convention in any new source.

Basketball Reference gives traded players multiple rows per season plus a `TOT` total
row; `data_cleaning.keep_tot_or_first` collapses these to one row per player-season.

## Modeling contract (read before editing modeling.py)

- **Target**: `fantasy_points_future` — created in `feature_engineering.py` as
  `groupby(player).fantasy_points.shift(-1)`. This is why the *last* season per player
  has a null target and why `split_data_nba` drops `year >= PRED_YEAR`.
- **Leakage is the primary hazard.** Features must describe a player's *history up to and
  including* the current season, never the future. `run_feature_pipeline`'s rolling and
  career aggregations are all backward-looking by construction. A prior feature
  (`years_before_peak`) was removed for leaking future information — be suspicious of any
  feature that could encode knowledge of the season being predicted.
- **Split discipline**: `split_data_nba` produces train/val/**test**. Test is a pure
  holdout — never used for tuning or early stopping. Val drives Hyperopt tuning and
  XGBoost early stopping. The final model (`create_model_nba`) refits on train+val
  combined, scaling `n_estimators` up by `n_estimators_mult` to compensate for the
  larger data.
- **Categoricals**: XGBoost runs with `enable_categorical=True`. `sanitize_dtypes`
  converts pandas `StringDtype` → object → `category` because XGBoost chokes on
  `string[python]`-backed categories. Run new categorical columns through it.
- **Hyperparameters**: search space is `config.SPACE` (leaf-based / `lossguide` growth);
  tuning uses Hyperopt TPE. Random seeds are passed explicitly everywhere for
  reproducibility — keep them.
- **Prediction intervals**: `generate_prediction_intervals` bootstraps at the *player*
  level (`group_col="player_id"`) and uses out-of-bag players for early stopping and
  residual sampling. Note the TODO flagging the current resampling methodology.

## Config

`config.py` centralizes tunables: league/roster settings and position splits
(`G_SPLIT`/`W_SPLIT`/`B_SPLIT`), the Hyperopt `SPACE`, column selections
(`SELECTED_COLUMNS`, `BREF_COLS`, `CORE_STATS`, `CAREER_STATS`), `NAME_OVERRIDES`,
award-parsing maps, and the plot `COLOR_PALETTE`. Prefer adding constants here over
hard-coding them in modules or the notebook.

## Plotting

`data_viz.py` holds all plots, built with **plotnine** (ggplot2 grammar) on the shared
`theme_nba()` and `COLOR_PALETTE`. Follow the existing layered `geom_*`/`aes()`/`labs()`
style when adding plots.

## Known TODOs (from notebook / commits)

- Scoring format is hard-coded to Underdog Fantasy in `calculate_fantasy_points`; make it
  league-configurable.
- FantasyPros scraping and position-blending functions still live inline in the notebook
  and should be moved into `.py` modules.
- Bootstrap resampling methodology needs revision (flagged inline in
  `generate_prediction_intervals`).
