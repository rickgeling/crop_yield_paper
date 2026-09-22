# Formerly: run_static_model_3.py

import os
import warnings
import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
from statsmodels.tsa.filters.hp_filter import hpfilter

import model_builder
import results_config as cfg
import importlib
importlib.reload(model_builder)

## FUNCTIONS DEPENDING ON THE MODEL AND THE (MISSING) DATA ##

def impute_mY(mY, years, method='linear'):
    """
    Impute missing in mY (shape T×N) by per-column interpolation
    on the year index, then forward/backward fill.
    """
    df = pd.DataFrame(mY, index=years)
    df = df.interpolate(method=method, axis=0) \
           .ffill(axis=0) \
           .bfill(axis=0)
    return df.values


def report_nans(mY, mX, mZ, regions):
    # mY: T×N
    nan_y = np.isnan(mY).any(axis=0)  # length N
    bad_y = [regions[i] for i, flag in enumerate(nan_y) if flag]
    print("mY: regions with NaNs:", bad_y or "None")

    # mX: N×T×d
    for dim in range(mX.shape[2]):
        nan_x = np.isnan(mX[:,:,dim]).any(axis=1)
        bad_x = [regions[i] for i, flag in enumerate(nan_x) if flag]
        print(f"mX (weather dim {dim}): regions with NaNs:", bad_x or "None")

    # mZ: N×T×(d+1)
    for dim in range(mZ.shape[2]):
        nan_z = np.isnan(mZ[:,:,dim]).any(axis=1)
        bad_z = [regions[i] for i, flag in enumerate(nan_z) if flag]
        print(f"mZ (regressor dim {dim}): regions with NaNs:", bad_z or "None")
        
        
def read_data(path, file_name):
    """
    Reads a CSV file into a pandas DataFrame, enforcing specific string data types.

    Parameters:
    ----------
    path : str
        The directory path where the file is located.
    file_name : str
        The name of the CSV file.

    Returns:
    -------
    pd.DataFrame
        The DataFrame read from the CSV file.
    """
    # Specify the columns that must be read as strings to preserve leading zeros.
    dtype_dict = {
        'county': str,
        'division_yield': str,
        'division_noaa': str
    }

    # Construct the full path to the file.
    full_path = os.path.join(path, file_name)
    
    # Read the CSV while enforcing the desired data types.
    df_model = pd.read_csv(full_path, dtype=dtype_dict)
    
    return df_model

HP_LAMBDA  = 6.25   # Ravn-Uhlig (2002) for annual data
YEAR_SPLIT = 1987   # first half 1951-1987, second half 1988-2024


def _matrices_to_long(mY, mX, years, regions, weather_cols):
    """Convert T×N mY and N×T×d mX to a long panel DataFrame."""
    records = []
    for n, county in enumerate(regions):
        state = county.split("_")[0]
        for t, yr in enumerate(years):
            row = {"county_id": county, "state": state, "year": int(yr), "lny": mY[t, n]}
            for j, col in enumerate(weather_cols):
                row[col] = mX[n, t, j]
            records.append(row)
    return pd.DataFrame(records)


def _hp_cycle(s):
    cycle, _ = hpfilter(s.astype(float), lamb=HP_LAMBDA)
    return cycle


def _fit_specA(data, weather_cols):
    """
    Spec A (Schlenker & Roberts 2009): county FE + state-specific quadratic
    time trend + weather anomalies, all in one regression.
    t is re-centered within the window to avoid collinearity.
    """
    d = data.copy()
    d["t"]  = d["year"] - d["year"].mean()
    d["t2"] = d["t"] ** 2

    state_d  = pd.get_dummies(d["state"], prefix="st", dtype=float)
    trend_t  = state_d.mul(d["t"].values,  axis=0).add_suffix("_t")
    trend_t2 = state_d.mul(d["t2"].values, axis=0).add_suffix("_t2")

    X = pd.concat([d[weather_cols], trend_t, trend_t2], axis=1)
    X.index = d.set_index(["county_id", "year"]).index
    lny = d.set_index(["county_id", "year"])["lny"]

    dat = pd.concat([lny, X], axis=1).dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        res = PanelOLS(
            dat["lny"], dat.drop(columns="lny"),
            entity_effects=True, drop_absorbed=True
        ).fit(cov_type="clustered", cluster_entity=True)
    return res


def _fit_specB(data, weather_cols):
    """
    Spec B (Ortiz-Bobea 2018): regress the HP-cycle of log yield on weather
    anomalies with county FE. HP filter must be applied to the full series
    before calling this function (already stored in 'lny_cycle').
    """
    panel = data.set_index(["county_id", "year"])
    dat = pd.concat([panel["lny_cycle"], panel[weather_cols]], axis=1).dropna()
    res = PanelOLS(
        dat["lny_cycle"], dat[weather_cols],
        entity_effects=True, drop_absorbed=True
    ).fit(cov_type="clustered", cluster_entity=True)
    return res


def _collect_results(res, weather_cols, window, spec):
    rows = []
    available = list(res.params.index)
    for v in weather_cols:
        if v not in available:
            print(f"  WARNING: '{v}' was absorbed/dropped in {spec} {window} — "
                  "check for collinearity with the trend block.")
            continue
        ci = res.conf_int()
        rows.append({
            "window": window, "spec": spec, "var": v,
            "coef":  res.params[v],
            "se":    res.std_errors[v],
            "p":     res.pvalues[v],
            "lo95":  ci.loc[v, "lower"],
            "hi95":  ci.loc[v, "upper"],
        })
    return rows


def run_static_benchmark_pipeline(
    df_master,
    model_id,
    season_id,
    config_options,
    states_str,
    output_dir=None,
):
    """
    Static benchmark pipeline: Spec A (trend-in-regression) and Spec B
    (HP-detrended outcome), each on full / first-half / second-half samples.

    Returns the long comparison DataFrame and saves it to output_dir.
    """
    output_dir = output_dir or cfg.static_dir()
    run_label = f"{states_str}_{model_id}_{season_id}_static"
    print("=" * 70)
    print(f"STARTING STATIC BENCHMARK: {run_label}")
    print("=" * 70)

    # --- 1. Prepare data (reuse existing pipeline for filtering / anomalies) ---
    mY, mX, mZ, years, regions, \
    var_names, ctrl_names, var_titles = \
        model_builder.model_specification(
            df_raw_input=df_master.copy(),
            model_type=model_id,
            season=season_id,
            config_options=config_options,
        )
    mY = impute_mY(mY, years)

    weather_cols = ctrl_names[1:]   # skip 'GT' — that's mZ[:,  :, 0], not in mX
    print(f"\nWeather regressors: {weather_cols}")

    # --- 2. Convert to long panel ---
    df_panel = _matrices_to_long(mY, mX, years, regions, weather_cols)

    # --- 3. HP-filter the FULL series once, then subset ---
    df_panel["lny_cycle"] = (
        df_panel.groupby("county_id")["lny"].transform(_hp_cycle)
    )
    print(f"HP filter applied (lambda={HP_LAMBDA}) to {df_panel['county_id'].nunique()} counties.")

    # --- 4. Fit all 6 models ---
    subsamples = {
        "full":   df_panel["year"].between(1951, 2024),
        "first":  df_panel["year"].between(1951, YEAR_SPLIT),
        "second": df_panel["year"].between(YEAR_SPLIT + 1, 2024),
    }

    all_rows = []
    for window, mask in subsamples.items():
        sub = df_panel[mask].copy()
        n_years = sub["year"].nunique()
        n_units = sub["county_id"].nunique()
        print(f"\n  [{window}]  {n_years} years x {n_units} counties")

        print(f"    Fitting Spec A (trend-in-regression)...")
        resA = _fit_specA(sub, weather_cols)
        all_rows += _collect_results(resA, weather_cols, window, "A_Schlenker")

        print(f"    Fitting Spec B (HP-detrended)...")
        resB = _fit_specB(sub, weather_cols)
        all_rows += _collect_results(resB, weather_cols, window, "B_Bobea")

    comparison = pd.DataFrame(all_rows)

    # --- 5. Print pivot of KDD and GDD across windows ---
    print("\n" + "=" * 70)
    print("COEFFICIENT COMPARISON  (Spec A | Spec B  x  full / first / second)")
    print("=" * 70)
    pivot = comparison.pivot_table(
        index=["spec", "var"], columns="window", values="coef"
    )[["full", "first", "second"]]
    print(pivot.to_string())

    # --- 6. Save ---
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{run_label}_coefs.csv")
    comparison.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")

    return comparison


# ── run ────────────────────────────────────────────────────────────────────────
df_model = read_data(os.path.dirname(cfg.master_panel_path()),
                     os.path.basename(cfg.master_panel_path()))
print(f"Loaded {cfg.master_panel_path()}. Shape: {df_model.shape}")
# NOTE: the panel is already filtered by pre_diagnostics/missing_data.py and no
# longer has the 2024 duplicate rows, so nothing is dropped here any more.

regions_to_exclude = ["27_031", "27_075"]
processing_type    = "anomaly"
all_codes_no_NE    = ["17", "18", "19", "27"]   # Illinois, Indiana, Iowa, Minnesota
states_str_no_NE   = "_".join(all_codes_no_NE)

config_no_NE = {
    'states_to_include':    all_codes_no_NE,
    'drop_region_names':    regions_to_exclude,
    'yield_processing':     {'log': True, 'detrend': False},
    'regressor_processing': processing_type,
    'model_a_squares':      'all',
    'model_1_squares':      'all',
}

comparison_df = run_static_benchmark_pipeline(
    df_master=df_model,
    model_id="model_3",
    season_id="GS",
    config_options=config_no_NE,
    states_str=states_str_no_NE,
    output_dir=cfg.static_dir(),
)

