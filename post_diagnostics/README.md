# post_diagnostics

Everything that reads a finished LLDVE run. Three folders, by what the output
is for:

- `econometrics/` - do the model's assumptions hold. Heteroskedasticity,
  serial correlation, cross-sectional dependence, effective sample size and
  bandwidth sensitivity. These justify the AWB bootstrap in the Methods.
- `figures/` - the paper's figures. The coefficient panel (Fig. 2), the
  temperature-yield response (Fig. 3), the GDD/KDD sample characterisation,
  and `temp_yield_response.ipynb`, the notebook where the response curves are
  explored per window and per state.
- `impact/` - what a weather shock is worth in yield terms.
  `2012_heat_impact.py` takes the 2012 anomaly and evaluates it against the
  sensitivity of different periods; `joint_dist_KDD.py` does the same
  percentile-based; `get_impact_projections.py` is the linear-trend version
  (Eq. 1 / Fig. 4); `get_X_change_stats.py` describes how the weather itself
  changed. One of the first three becomes the paper's framing, that choice is
  still open.

## which run they read

None of these scripts has a path of its own. They all read
`results_config.py` in the repo root:

```python
DATASET = "corn"                           # corn | soy | corn_east100m
RUN_TAG = "17_model_3_GS_logY_anomalyX"    # folder under paper_main_results/<DATASET>/
YEAR_MIN, YEAR_MAX = 1951, 2024
```

Change those two lines and every script here follows. `python results_config.py`
prints what is currently selected and whether that run exists.

The paper's main run will be `17_18_19_27_model_3_GS_logY_anomalyX`; at the
moment only the Illinois test run exists.

## status

- `figures/` and `impact/` are repointed and ready, though only checked as far
  as the paths go, not rerun yet.
- `econometrics/` is moved but not repaired. All five scripts still point at
  result folders that no longer exist, two of them use model 2 instead of
  model 3, and `bandwith_sensitivity.py` imports `methods.LLDVE_new`, which
  does not exist. Fix these once the real runs are in.
