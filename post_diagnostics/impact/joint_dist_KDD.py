"""
KDD extreme-heat analysis (replaces the mean-trend projection).

The mean KDD trend is a weak signal in a spiky series, and its sign flips
across estimation windows.  KDD damage does not arrive as a smooth drift; it
arrives in occasional severe years.  A linear trend through that series throws
away the thing that does the damage.  This script asks the two questions the
mean trend hides.

PART A — Empirical tail comparison (assumption-free).
  Has the DISTRIBUTION of county-level KDD shocks changed between the early
  (1951-1987) and recent (1988-2024) periods?  Compare upper percentiles, the
  variance, and the frequency of county-years above the early-period 95th
  percentile.  A heavier recent tail means heat risk has shifted into the
  extremes even if the mean is flat.

PART B — Severe-year scenario through the model (KDD only, no net impact).
  Define a representative severe-heat shock as the recent-period 95th-percentile
  county-level KDD anomaly.  Hold that shock fixed and evaluate the implied
  yield hit at CURRENT KDD sensitivity vs PEAK-ERA KDD sensitivity.  The
  difference isolates the documented heat-tolerance recovery: are we better
  protected against a bad heat year now than at the KDD-damage peak?

GDD is intentionally excluded: the scenario conditions on a heat extreme and is
a protection-against-heat statement, not a net yield forecast.
"""

import os
import numpy as np
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────────
# ── paths (dataset and run are set in results_config.py at the repo root) ─────
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import results_config as cfg

BASE      = cfg.run_dir()
CI_DIR    = cfg.ci_dir()
MODEL_TAG = cfg.RUN_TAG

YEARS       = np.arange(1951, 2025)   # 74 years, matches T in saved arrays
EARLY_END   = 1987                    # early period 1951-1987
RECENT_START = 1988                   # recent period 1988-2024
ALPHA       = 0.05


# ── helpers ────────────────────────────────────────────────────────────────────

def load_coef(var: str) -> pd.Series:
    fname = f"{var}_ST_PW_{MODEL_TAG}.csv"
    df    = pd.read_csv(os.path.join(CI_DIR, fname))
    return pd.Series(df["Beta_hat"].values, index=YEARS)


def five_yr_end(path: pd.Series) -> float:
    """Current sensitivity: mean of final 5 years (2020-2024)."""
    return path.iloc[-5:].mean()


def avg_window(path: pd.Series, start: int, end: int) -> float:
    return path.loc[start:end].mean()


def rolling_window_peak(series: pd.Series, mode: str, window: int = 5):
    """(start, end, value) for the 5-yr rolling window with the most negative
       (mode='min') or highest (mode='max') average.  Index label = last year."""
    roll = series.rolling(window).mean().dropna()
    pk   = roll.idxmin() if mode == "min" else roll.idxmax()
    return pk - window + 1, pk, roll[pk]


def to_pct(dln: float) -> float:
    """Exact log-to-percent."""
    return (np.exp(dln) - 1) * 100


def to_logpoints(dln: float) -> float:
    """Approximate percent (log points)."""
    return dln * 100


# ── load coefficient path + bootstrap draws ────────────────────────────────────
kdd_path = load_coef("KDDGS_anom")

# bootstrap draws: (B=1499, T=74, n_coefs=6); KDD is index 2
draws_all = np.load(os.path.join(BASE, "mTheta_stars_draws.npy"))
assert draws_all.shape[1] == len(YEARS), \
    f"T mismatch: draws have {draws_all.shape[1]} years, YEARS has {len(YEARS)}"
B        = draws_all.shape[0]
kdd_boot = draws_all[:, :, 2]   # (B, T)

# ── load county-year KDD anomalies: mX (N=334, T=74, d=5); KDD is index 1 ──────
mX_raw = np.load(os.path.join(BASE, "input_mX.npy"))
kdd_cy = mX_raw[:, :, 1]        # (N, T) county-year KDD anomaly


# ══════════════════════════════════════════════════════════════════════════════
# PART A — EMPIRICAL TAIL COMPARISON  (county-year pooled, assumption-free)
# ══════════════════════════════════════════════════════════════════════════════
early_mask  = YEARS <= EARLY_END
recent_mask = YEARS >= RECENT_START

kdd_early  = kdd_cy[:, early_mask].ravel()
kdd_early  = kdd_early[~np.isnan(kdd_early)]
kdd_recent = kdd_cy[:, recent_mask].ravel()
kdd_recent = kdd_recent[~np.isnan(kdd_recent)]

# threshold = early-period 95th pct; under stationarity recent exceedance ~ 5%
thr = np.percentile(kdd_early, 95)


def tail_summary(s: np.ndarray, threshold: float) -> dict:
    return {
        "n":            len(s),
        "mean":         s.mean(),
        "sd":           s.std(),
        "p90":          np.percentile(s, 90),
        "p95":          np.percentile(s, 95),
        "p99":          np.percentile(s, 99),
        "max":          s.max(),
        "frac>thr (%)": (s > threshold).mean() * 100,
    }


tail = pd.DataFrame({
    f"early 1951-{EARLY_END}":   tail_summary(kdd_early,  thr),
    f"recent {RECENT_START}-2024": tail_summary(kdd_recent, thr),
}).T

print("=" * 70)
print("PART A — KDD anomaly distribution: early vs recent (county-year pooled)")
print("=" * 70)
print(f"Threshold = early-period 95th pct = {thr:.1f} KDD-anomaly degree-days")
print("(if stationary, recent 'frac>thr' should be ~5%; higher = heavier tail)")
print()
print(tail.round(2).to_string())


# ══════════════════════════════════════════════════════════════════════════════
# PART B — SEVERE-YEAR SCENARIO  (KDD only; current vs peak-era sensitivity)
# ══════════════════════════════════════════════════════════════════════════════
# Fixed severe-heat shock: recent-period 95th-pct county-year KDD anomaly.
shock = np.percentile(kdd_recent, 95)

# KDD sensitivity at two epochs (peak window from the point estimate, fixed for
# all bootstrap draws to avoid winner's-curse bias on the extremum).
kdd_ps, kdd_pe, _ = rolling_window_peak(kdd_path, mode="min")
b_now  = five_yr_end(kdd_path)
b_peak = avg_window(kdd_path, kdd_ps, kdd_pe)

dmg_now_ln  = b_now  * shock     # log-yield hit from the severe shock, now
dmg_peak_ln = b_peak * shock     # ... at peak-era sensitivity
protect_ln  = dmg_now_ln - dmg_peak_ln   # >0 means less damage now (recovery)

# within-replicate CIs: differencing inside each draw preserves the correlation
# between the two epochs (same coefficient path), giving a tight, honest band.
now_draws  = np.array([five_yr_end(pd.Series(kdd_boot[i], index=YEARS)) * shock
                       for i in range(B)])
peak_draws = np.array([avg_window(pd.Series(kdd_boot[i], index=YEARS),
                                  kdd_ps, kdd_pe) * shock
                       for i in range(B)])
protect_draws = now_draws - peak_draws

def ci(arr):
    return np.percentile(arr, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])

now_lo,  now_hi  = ci(now_draws)
peak_lo, peak_hi = ci(peak_draws)
prot_lo, prot_hi = ci(protect_draws)

print()
print("=" * 70)
print("PART B — Implied yield hit from a severe heat year (KDD only)")
print("=" * 70)
print(f"Severe-heat shock = recent 95th-pct county KDD anomaly = "
      f"{shock:.1f} degree-days  (held fixed across epochs)")
print(f"KDD-peak window (most negative 5-yr avg): {kdd_ps}-{kdd_pe}")
print()
print(f"  Sensitivity at peak {kdd_ps}-{kdd_pe}:  beta_KDD = {b_peak:+.5f}")
print(f"  Sensitivity now     2020-2024:  beta_KDD = {b_now:+.5f}")
print()
print(f"Implied yield hit from the same severe shock:")
print(f"  at peak-era sensitivity:  {to_logpoints(dmg_peak_ln):+.2f} log pts  "
      f"[{to_logpoints(peak_lo):+.2f}, {to_logpoints(peak_hi):+.2f}]")
print(f"  at current sensitivity:   {to_logpoints(dmg_now_ln):+.2f} log pts  "
      f"[{to_logpoints(now_lo):+.2f}, {to_logpoints(now_hi):+.2f}]")
print(f"  protection gained:        {to_logpoints(protect_ln):+.2f} log pts  "
      f"[{to_logpoints(prot_lo):+.2f}, {to_logpoints(prot_hi):+.2f}]")
print()
print(f"Headline (exact %): a severe heat year implies "
      f"{to_pct(dmg_now_ln):+.2f}% now vs {to_pct(dmg_peak_ln):+.2f}% at peak.")
print("Protection gained is significant if the CI above excludes zero.")