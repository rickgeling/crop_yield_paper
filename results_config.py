"""
Where the post-estimation scripts read their results from.

One place to set the dataset and the run, so switching from corn to soy, or
from one state to the pooled run, is a single edit here instead of twelve edits
spread over post_diagnostics/.

Scripts in post_diagnostics/<subfolder>/ pick this up with:

    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    import results_config as cfg

    BASE      = cfg.run_dir()
    CI_DIR    = cfg.ci_dir()
    MODEL_TAG = cfg.RUN_TAG

Scripts in the repo root can just `import results_config as cfg`.
"""

import os

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- USER SETTINGS -----------------------------------------------------------

DATASET = "corn"#"soy"#"corn_east100m"

# Folder name under paper_main_results/<DATASET>/. The paper's main run is
# "17_18_19_27_model_3_GS_logY_anomalyX" (four rainfed states pooled), Nebraska
# is "31_model_3_GS_logY_anomalyX". Right now only the Illinois test run exists.
RUN_TAG = "17_model_3_GS_logY_anomalyX"

# Model window. Must match END_YEAR in the lldve_run notebooks, otherwise the
# YEARS axis here does not line up with the rows of mTheta_hat.
YEAR_MIN, YEAR_MAX = 1951, 2024

ALPHA = 0.05   # significance level for the bootstrap CIs

# -----------------------------------------------------------------------------

RESULTS_BASE = os.path.join(REPO_ROOT, "paper_main_results")
YEARS = np.arange(YEAR_MIN, YEAR_MAX + 1)

# Column order in mTheta_hat.csv and mTheta_stars_draws.npy for model 3.
# Also the file-name stems of the coefficient CSVs.
COEF_ORDER = ["GT", "GDDGS_anom", "KDDGS_anom",
              "PrecGS_anom", "PrecSqGS_anom", "KDDxPrecGS_anom"]

# Column order in input_mX.npy for model 3 (mX has no intercept, so GT is absent)
MX_ORDER = ["GDD", "KDD", "Prec", "PrecSq", "KDDxPrec"]


def run_dir(tag=None, dataset=None):
    """Folder of one run, e.g. paper_main_results/corn/17_model_3_GS_logY_anomalyX."""
    return os.path.join(RESULTS_BASE, dataset or DATASET, tag or RUN_TAG)


def ci_dir(tag=None, dataset=None):
    """Where estCI wrote the per-coefficient CSVs with the confidence bands."""
    return os.path.join(run_dir(tag, dataset), "coefficient_CIs_CSV")


def static_dir(dataset=None):
    """Output folder of the static benchmark (run_static_baseline.py)."""
    return os.path.join(RESULTS_BASE, dataset or DATASET, "static")


def master_panel_path(dataset=None):
    return os.path.join(REPO_ROOT, f"master_panel_{dataset or DATASET}.csv")


# --- loaders -----------------------------------------------------------------

def load_ci(var, tag=None, dataset=None):
    """One coefficient's estimate and bands: Beta_hat, Simult_LB/UB, Pointwise_LB/UB."""
    tag = tag or RUN_TAG
    path = os.path.join(ci_dir(tag, dataset), f"{var}_ST_PW_{tag}.csv")
    return pd.read_csv(path)


def load_theta(tag=None, dataset=None):
    """Estimated coefficients, shape (T, 6), columns in COEF_ORDER."""
    return np.loadtxt(os.path.join(run_dir(tag, dataset), "mTheta_hat.csv"), delimiter=",")


def load_draws(tag=None, dataset=None):
    """Bootstrap draws, shape (B, T, 6)."""
    return np.load(os.path.join(run_dir(tag, dataset), "mTheta_stars_draws.npy"))


def load_mX(tag=None, dataset=None):
    """Regressors as they went into the model, shape (N, T, 5), order MX_ORDER."""
    return np.load(os.path.join(run_dir(tag, dataset), "input_mX.npy"))


def load_regions(tag=None, dataset=None):
    return pd.read_csv(os.path.join(run_dir(tag, dataset), "regions_list.csv"))["region_name"].tolist()


def load_master(dataset=None):
    """The master panel this run was built from."""
    return pd.read_csv(master_panel_path(dataset),
                       dtype={'fips_full': str, 'county': str,
                              'division_yield': str, 'division_noaa': str})


def describe():
    """Print what is currently selected, and whether it exists on disk."""
    d = run_dir()
    print(f"dataset   : {DATASET}")
    print(f"run tag   : {RUN_TAG}")
    print(f"run dir   : {d}  {'(exists)' if os.path.isdir(d) else '(MISSING)'}")
    print(f"window    : {YEAR_MIN}-{YEAR_MAX}  (T = {len(YEARS)})")
    print(f"panel     : {master_panel_path()}")


if __name__ == '__main__':
    describe()
