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

**Pandas 3 string dtype:** `config.py` sets `pd.set_option("future.infer_string", False)`
on import. Pandas 3.0 defaults text columns to the new `str` dtype, which the Positron Data
Explorer can't apply text filters to (filters silently no-op); reverting to classic `object`
keeps the viewer usable. Don't remove it unless the workflow moves off Positron. It's set in
`config.py` so it applies before any DataFrame is built, in the notebook or a script.

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
   `fantasy_points` target from a scoring map (`config.SCORING_MAPS`, e.g. `"UD"`; selected
   as `SCORING_MAP` at the top of the notebook and reused for FantasyPros projections), then
   `feature_engineering.run_feature_pipeline` chains every feature step and produces the
   modeling matrix.
5. **Modeling** — `modeling.split_data_nba` → `tune_xgb_nba` → `create_model_nba`, with
   `create_baseline_nba` as a reference point and `generate_prediction_intervals` for
   bootstrapped ranges.
6. **Prediction + draft board** — predictions for `PRED_YEAR` are blended with scraped
   FantasyPros projections, assigned position groups, and ranked into a draft board.
   These later stages currently live as inline functions in the notebook (see TODOs).
7. **Value tiering** — after `relative_value` is computed, players are bucketed into value
   tiers by KMeans (elbow method to pick `k`): `modeling.segment_players` builds an overall
   `player_value_tier` across all players, and `modeling.segment_players_by_group` builds a
   `position_value_tier` within each `position_group` (per-group `k`). Both relabel clusters
   so tier 1 = highest value. `data_viz.plot_elbow` renders the WSS-vs-`k` elbow (plotnine).
   The final board carries both tiers.

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
  converts pandas string/object columns → `category`. It is called **once** on the full
  feature matrix inside `split_data_nba` *before* the train/val/test split — do not move it
  after the split. Under pandas 3, concatenating two `category` columns with mismatched
  category sets silently collapses them to a `str` dtype XGBoost rejects; sanitizing before
  the split gives every split identical category sets so the downstream `pd.concat` calls
  (baseline, final model, bootstrap) survive.
- **Prediction categoricals**: `build_prediction_frame` must be passed `train_features=X_train`
  so `align_categorical_dtypes` re-casts `X_pred`'s categoricals to the *training* category
  sets. XGBoost 3.x errors on categories unseen during training; alignment maps them to `NaN`
  (treated as missing). Omitting `train_features` reintroduces that error on the prediction path.
- **Hyperparameters**: search space is `config.SPACE` (leaf-based / `lossguide` growth);
  tuning uses Hyperopt TPE. Random seeds are passed explicitly everywhere for
  reproducibility — keep them.
- **Prediction intervals**: `generate_prediction_intervals` takes the already-fit final
  `model` (call it *after* `create_model_nba`, not before) and runs a player-level cluster
  bootstrap — players sampled with replacement, each drawn player's rows replicated by its
  draw count, refit at the model's fixed tree count with no early stopping. Bands center on
  the final-model prediction (the draft-sheet number); width combines refit spread
  (epistemic) with a global out-of-bag residual pool binned by fitted value (heteroscedastic
  aleatoric noise). Outputs `pred_p05..p95` plus `ceiling_index`/`floor_index`/`upside_index`
  (standardized league-wide across all predicted players, mean 100/sd 15; `upside_index` is
  an OR-score that leans ceiling by design).
- **`n_bootstrap=30` is the confirmed default** — do not lower it casually. It is a
  deliberate reliability/variance tradeoff: the low count keeps run-to-run churn (a fresh
  `random_state` gives a genuinely fresh simulation, so you don't over-index on the same
  flagged players). Empirically (6-seed sweeps, model held fixed): cross-seed rank
  correlation ≈ 0.62 (nb=12) → 0.73 (nb=20) → 0.77 (nb=30) for `ceiling_index`; the 20→30
  gain is small (diminishing returns) and the highlighted top-30 rank churn is unchanged,
  while 30 gives a steadier mid-board. Below ~20 the estimate gets noisy (`upside_index`
  ρ≈0.51 at nb=12). 30 sits at the reliability/variance knee.

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

**Two deliberate Plotly exceptions:** `plot_positional_tiers` and `plot_model_vs_expert`
(the final-step tier/rank diagnostics) are built with **Plotly**, not plotnine, because they
need hover-to-identify-player to be legible (10-14 tier colors + a label per point). They
return a `plotly.graph_objects.Figure` and color tiers with a MetBrewer "Hiroshige" ramp
(`config.HIROSHIGE_COLORS`) interpolated to the tier count via `data_viz.tier_palette()`
(mizani's `gradient_n_pal`, which ships with plotnine). Everything else stays plotnine.

**Positron display gotcha:** tall/inline plotnine figures should be shown with `display(fig)`,
not `fig.show()` — `.show()` routes to Positron's fixed-size Plots pane and crops them (see
`plot_pred_vs_proj_dumbbell` usage). Plotly figures render fine inline via `.show()`.

## Known TODOs (from notebook / commits)

- FantasyPros scraping and position-blending functions still live inline in the notebook
  and should be moved into `.py` modules.
