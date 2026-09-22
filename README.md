# crop_yield_paper

The modelling side of the maize weather-sensitivity paper. It takes the
county-year panels built in
[EpiNOAA-Python](https://github.com/rickgeling/EpiNOAA-Python), turns them into
model-ready master panels, and estimates time-varying coefficient models on top
of them.

The paper says maize, the code says corn. Same crop, I've left the code alone
rather than renaming every file and variable.

## paper

Uneven buffering of U.S. maize yields against extreme heat - a time-varying
coefficient panel approach

> U.S. maize production supplies one-third of the global harvest and faces
> escalating risks from extreme weather. Despite decades of technological
> advances intended to enhance climate resilience, how large-scale yield
> sensitivity to temperature and rainfall has evolved remains poorly
> understood.

Link and full citation once available.

## why

The data repo stops at "county-year rows with yield and weather attached". Two
things still have to happen before a model can run: decide which counties are
complete enough to use, and express the weather as anomalies rather than
levels. That's `pre_diagnostics/`, and its output is the master panel every
later step reads.

The estimator itself is LLDVE, a local-linear dummy-variable estimator for
panels with time-varying coefficients. It lets every weather coefficient move
over time instead of forcing one number on 74 years, which is the whole point
of the paper: sensitivity to heat isn't constant.

## setup

Python 3.12 with numpy, pandas, scipy, matplotlib, seaborn, joblib and
statsmodels. Two extras that aren't needed for the main runs:

- `linearmodels`, only for `run_static_baseline.py`. Version 6.1 doesn't work
  with numpy 2 (`np.unicode_` was removed), so upgrade it before running that
  script.
- `geopandas`, only for the county map in `pre_diagnostics/all_pre_diagnostics.ipynb`.

Notebooks run interactively in VS Code. Everything is path-relative to the repo
root, so run scripts from there: `python post_diagnostics/impact/2012_heat_impact.py`.

## layout

At the root:

- `model_builder.py` turns a master panel into the arrays a model needs
  (`mY`, `mX`, `mZ`), one builder per model specification.
- `lldve_runs.py` is what the three run notebooks share: reading a panel,
  the county check, imputation, one-run wrapper, coefficient plots.
- `results_config.py` says which run the post-estimation scripts read:
  dataset, run tag and year window. One edit there moves all of them.
  `python results_config.py` prints what's currently selected.
- `run_static_baseline.py` is the static panel benchmark the time-varying
  results are anchored against.
- `lldve_run_corn.ipynb`, `lldve_run_soy.ipynb`, `lldve_run_corn_east100m.ipynb`
  are the run notebooks, one per dataset.

Folders:

- `pre_diagnostics/` builds the master panels. Also holds the importer files
  copied over from the data repo, and `all_pre_diagnostics.ipynb` for
  descriptive statistics and the yield map.
- `methods/` holds the estimators: LLDVE and the restricted cubic spline it is
  cross-validated against.
- `post_diagnostics/` is everything that reads a finished run, split into
  `econometrics/` (do the assumptions hold), `figures/` (the paper's figures)
  and `impact/` (what a weather shock is worth). See it's own README.
- `paper_main_results/` is where runs are written, `thesis_results/` holds the
  thesis runs the new ones were checked against.
- `archive/` is everything no longer live: `tested_not_used/` for analyses run
  on purpose but not the paper's choice, `to_delete/` for retired files, and
  `cleanup_notes/` for the notes from the 2026 clean-up.

Data, results and the archive are not in git, they're rebuildable and large.
My own notes live in `x_logbook/`, also not in git.

## the pipeline

Per dataset, run in this order:

1. Copy the importer file from
   `EpiNOAA-Python/03b_weather_nclimgrid_importer/created_dfs_step_final/`
   into `pre_diagnostics/`.
2. `pre_diagnostics/missing_data.py` applies the county filter: count, per
   county, how many years in the window have a gap in any key column, and drop
   the county entirely if that's more than `MISSING_YEAR_TOLERANCE` (4, the
   Zipper et al. 2016 rule). Settings sit at the top of the file.
3. `pre_diagnostics/anomalies.py` reads the filtered file and writes the master
   panel one level up. It builds `fips_full` as `17_001`, creates `log_value`,
   and subtracts each county's 1981-2010 mean from every weather variable to
   get `<var>_anom`, plus `<var>_anom_sq` for the quadratic terms. A fixed
   baseline, so a change in estimated sensitivity isn't an artefact of a moving
   reference. No detrending: the anomaly has a clear interpretation, a
   statistical trend fit doesn't.
4. `lldve_run_<dataset>.ipynb` estimates the models. Settings in one cell at
   the top, the runs in a `RUNS` dict below it.
5. `post_diagnostics/` reads the results.

Both pre-diagnostics scripts append to `pre_diagnostics/panel_log.json`: when,
which file, which settings, how many counties. The run notebooks look up the
last entry for their panel and refuse to run if the window doesn't match their
own, so a panel filtered on 1951-2024 can't silently be modelled on 1951-2025.

## data

Three master panels, all filtered on 1951-2024 with tolerance 4:

| panel | crop | counties | rows |
|---|---|---|---|
| `master_panel_corn.csv` | corn | 408 (334 rainfed + 74 Nebraska) | 39,576 |
| `master_panel_soy.csv` | soy | 335 | 33,165 |
| `master_panel_corn_east100m.csv` | corn, east of the 100th meridian | 588 | 68,208 |

The panels keep every year of the counties that survive the filter; the year
window is applied in the run notebooks. Gaps in yield are interpolated there,
per county, not here.

How the underlying data is built, and what is known to be wrong with it, is
documented in [EpiNOAA-Python](https://github.com/rickgeling/EpiNOAA-Python).

## the model

Model 3, the paper's specification: log yield on GDD, KDD, precipitation,
precipitation squared and a KDD x precipitation interaction, all as anomalies,
all over the growing season (1 April to 30 September). Every coefficient is a
smooth function of time. Inference is by autoregressive wild bootstrap, which
is what the residual diagnostics call for.

The run tag is `<states>_<model>_<season>_logY_<processing>X`, so the paper's
main run is `17_18_19_27_model_3_GS_logY_anomalyX`: Illinois, Indiana, Iowa and
Minnesota pooled. Nebraska (31) always runs on its own, as the irrigation
benchmark against the rain-fed four.

A run folder holds the estimated coefficients (`mTheta_hat.csv`), the fixed
effects, the bandwidth actually used, fitted values and residuals, the
bootstrap draws, a CSV per coefficient with pointwise and simultaneous bands,
and the R-squared and explained-variation output.

## where things stand

- All three panels are current, rebuilt September 2026.
- Corn and soy are estimated for the pooled four states and for Illinois on its
  own; east of the 100th meridian is estimated pooled. All at B = 149, with the
  bandwidths the thesis found by PLMCV.
- The pooled corn run reproduces the thesis exactly: same 334 counties,
  coefficients differing by at most 7e-5. The rebuilt pipeline is confirmed.
- Still to run: the remaining single states, Nebraska, and everything again at
  B = 1499 for the final numbers.
- `post_diagnostics/figures/` and `impact/` follow `results_config.py`;
  `econometrics/` points at the pooled corn run but hasn't been run yet.

## notes

- Bandwidth is currently set by hand, to the values PLMCV picked in the thesis
  (0.26 to 0.34 depending on the run). Wether the final runs use PLMCV again
  is still open.
- The 100th meridian dataset is missing most of Illinois, Indiana and Iowa: 11,
  10 and 8 counties instead of 102, 92 and 99. Those states lie entirely east
  of -100, so it isn't the geographic filter, it's a truncated USDA Quick Stats
  download in the data repo. Needs re-downloading before that branch means
  anything for the Corn Belt.
- July 2023 precipitation in nClimGrid-Daily is too low, which carries into
  `PREC_GS` and `CHD_GS` for that year. The problem is in NOAA's own files, not
  in this code. How to handle it in the paper is still open; the data is
  untouched for now. Details in the EpiNOAA-Python repo.
- 2025 is in the panels but incomplete on the yield side, roughly 310 of 470
  corn counties as of the August 2026 USDA export. That's why the window stops
  at 2024. Moving it is one setting in `missing_data.py` and one in each run
  notebook, and they check each other.
- The precipitation extremes (CDD, Rx5day, R10mm, R20mm) aren't in the current
  importer files; they come from a seperate importer variant. The column names
  are still listed in both pre-diagnostics scripts, so dropping in a file that
  has them is enough to get their anomalies back. Models 4, 5 and 7 need them.
- `CHD_GS` and `CHD_SGF` do get anomalies, although no model uses them yet.

## citation

BibTeX, once available.
