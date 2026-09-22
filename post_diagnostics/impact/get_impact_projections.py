"""
Impact projections — Equation (1).

  Δln(Yield)_2054 = ΔGDD_2024-2054 × β̂_GDD,2024
                  + ΔKDD_2024-2054 × β̂_KDD,2024

β̂_2024  : 5-year end-of-sample mean of the LLDVE path (2020–2024), for
            boundary stability.
ΔGDD/KDD : fitted linear trend in the pooled county-mean anomaly over
            1994–2024, multiplied by 30.  The slope uses all 31 years so it
            is not driven by single-season end-points.

Precipitation and the compound term are excluded: projecting their product
would compound uncertainty and sits awkwardly with the deliberately simple
framing of the exercise.

CI: AWB bootstrap replicates pushed through the same calculation with
    weather trends fixed (observed data, not estimated quantities).  The
    band is a statement about sensitivity uncertainty alone.

Decomposition for Figure 4 is in log points (100 × Δln) so GDD and KDD bars
stack exactly to the net.  The exact 100(exp(Δln) − 1) transform is used
only for the headline net figure in the prose.
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

YEARS   = np.arange(1951, 2025)   # 74 years — matches T in all saved arrays
Y0, Y1  = 1994, 2024              # 31-year trend window
HORIZON = 30
ALPHA   = 0.05


# ── helpers ────────────────────────────────────────────────────────────────────

def load_coef(var: str) -> pd.Series:
    fname = f"{var}_ST_PW_{MODEL_TAG}.csv"
    df    = pd.read_csv(os.path.join(CI_DIR, fname))
    return pd.Series(df["Beta_hat"].values, index=YEARS)


def five_yr_end(path: pd.Series) -> float:
    """Frozen end-of-sample sensitivity: mean of final 5 years."""
    return path.iloc[-5:].mean()


def trend_change(anom: pd.Series, y0=Y0, y1=Y1, horizon=HORIZON) -> float:
    """Projected change = fitted per-year slope over [y0, y1] × horizon."""
    s     = anom.loc[y0:y1]
    slope = np.polyfit(s.index.values, s.values, 1)[0]
    return slope * horizon


def to_pct(dln: float) -> float:
    """Exact log-to-percent.  Use for the NET headline only."""
    return (np.exp(dln) - 1) * 100


def to_logpoints(dln: float) -> float:
    """Approximate percent (log points).  Additive across components."""
    return dln * 100


def implied_net_ln(b_gdd, b_kdd, d_gdd, d_kdd):
    return b_gdd * d_gdd + b_kdd * d_kdd


# ── load coefficient paths ─────────────────────────────────────────────────────
gdd_path = load_coef("GDDGS_anom")
kdd_path = load_coef("KDDGS_anom")

# ── load bootstrap draws: shape (B=1499, T=74, n_coefs=6) ─────────────────────
# Coefficient order: [GT(0), GDD(1), KDD(2), Prec(3), PrecSq(4), KDDxPrec(5)]
draws_all = np.load(os.path.join(BASE, "mTheta_stars_draws.npy"))
assert draws_all.shape[1] == len(YEARS), \
    f"T mismatch: draws have {draws_all.shape[1]} years, YEARS has {len(YEARS)}"
B = draws_all.shape[0]

gdd_boot = draws_all[:, :, 1]   # (B, T)
kdd_boot = draws_all[:, :, 2]   # (B, T)

# ── load pooled weather anomalies: shape (N=334, T=74, d=5) ───────────────────
# Variable order in mX: [GDD(0), KDD(1), Prec(2), PrecSq(3), KDDxPrec(4)]
mX_raw   = np.load(os.path.join(BASE, "input_mX.npy"))
gdd_anom = pd.Series(np.nanmean(mX_raw[:, :, 0], axis=0), index=YEARS)
kdd_anom = pd.Series(np.nanmean(mX_raw[:, :, 1], axis=0), index=YEARS)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  POINT ESTIMATES
# ══════════════════════════════════════════════════════════════════════════════
b_gdd = five_yr_end(gdd_path)
b_kdd = five_yr_end(kdd_path)
d_gdd = trend_change(gdd_anom)
d_kdd = trend_change(kdd_anom)

contrib_gdd_ln = b_gdd * d_gdd
contrib_kdd_ln = b_kdd * d_kdd
net_ln         = contrib_gdd_ln + contrib_kdd_ln

print("=" * 62)
print("IMPACT PROJECTIONS  (pooled rainfed Corn Belt, Eq. 1)")
print("=" * 62)
print(f"Frozen sensitivities (5-yr avg 2020-2024):")
print(f"  beta_GDD = {b_gdd:+.5f}")
print(f"  beta_KDD = {b_kdd:+.5f}")
print(f"Projected weather change (1994-2024 slope x {HORIZON} yr):")
print(f"  delta_GDD = {d_gdd:+.4f}  degree-day anomaly")
print(f"  delta_KDD = {d_kdd:+.4f}  degree-day anomaly")
print()
print(f"Contributions (log points, additive):")
print(f"  GDD: {to_logpoints(contrib_gdd_ln):+.3f} log pts")
print(f"  KDD: {to_logpoints(contrib_kdd_ln):+.3f} log pts")
print(f"  Net: {to_logpoints(net_ln):+.3f} log pts")
print(f"  Net: {to_pct(net_ln):+.2f}%  (exact 100*[exp(dln)-1])")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  CONFIDENCE INTERVAL  (weather trends fixed; sensitivity varies)
# ══════════════════════════════════════════════════════════════════════════════
draws_ln = np.array([
    implied_net_ln(
        five_yr_end(pd.Series(gdd_boot[i], index=YEARS)),
        five_yr_end(pd.Series(kdd_boot[i], index=YEARS)),
        d_gdd, d_kdd,
    )
    for i in range(B)
])

lo_ln, hi_ln = np.percentile(draws_ln, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])

print()
print(f"95% CI (sensitivity uncertainty only; weather trend fixed):")
print(f"  Net: {to_pct(net_ln):+.2f}%  "
      f"[{to_pct(lo_ln):+.2f}, {to_pct(hi_ln):+.2f}]")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  DECOMPOSITION TABLE  (Figure 4 inputs)
# ══════════════════════════════════════════════════════════════════════════════
gdd_ln_draws = np.array([
    five_yr_end(pd.Series(gdd_boot[i], index=YEARS)) * d_gdd for i in range(B)
])
kdd_ln_draws = np.array([
    five_yr_end(pd.Series(kdd_boot[i], index=YEARS)) * d_kdd for i in range(B)
])

decomp = pd.DataFrame(
    {
        "GDD": (to_logpoints(contrib_gdd_ln),
                *np.percentile(to_logpoints(gdd_ln_draws), [2.5, 97.5])),
        "KDD": (to_logpoints(contrib_kdd_ln),
                *np.percentile(to_logpoints(kdd_ln_draws), [2.5, 97.5])),
        "Net": (to_logpoints(net_ln),
                *np.percentile(to_logpoints(draws_ln),     [2.5, 97.5])),
    },
    index=["est", "lo95", "hi95"],
).T

print()
print("Decomposition table (log points; GDD + KDD stack exactly to Net):")
print(decomp.to_string(float_format=lambda x: f"{x:+.3f}"))
print()
print("Note: quote Net in prose as exact % = "
      f"{to_pct(net_ln):+.2f}% [{to_pct(lo_ln):+.2f}, {to_pct(hi_ln):+.2f}];  "
      "quote GDD/KDD in log points only (bars are additive, exact % are not).")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  ROBUSTNESS: trend sensitivity to window choice
#
#  Two questions:
#    (a) Start-year sensitivity (end fixed at 2024): does starting in the GDD
#        trough (1994) inflate the slope?
#    (b) Window-length / placement: do shorter or shifted windows agree?
#
#  All rates are per decade so 20-year and 30-year windows are directly
#  comparable.  The headline projection (1994-2024, ×30) is unchanged; this
#  section only tests whether that choice is robust.
# ══════════════════════════════════════════════════════════════════════════════

def trend_per_decade(anom: pd.Series, y0: int, y1: int) -> float:
    """Fitted slope over [y0, y1] expressed per decade. NaN if < 5 obs."""
    s = anom.loc[y0:y1].dropna()
    if len(s) < 5:
        return np.nan
    return np.polyfit(s.index.values, s.values, 1)[0] * 10


# (label, start, end)
WINDOWS = [
    ("1994-2024  30y  headline", 1994, 2024),
    ("1990-2024  35y",           1990, 2024),
    ("1984-2024  40y",           1984, 2024),
    ("2005-2024  20y  recent",   2005, 2024),
    ("1995-2014  20y  shifted",  1995, 2014),
]

robust = pd.DataFrame(
    {
        label: {
            "GDD /decade": trend_per_decade(gdd_anom, y0, y1),
            "KDD /decade": trend_per_decade(kdd_anom, y0, y1),
        }
        for label, y0, y1 in WINDOWS
    }
).T

print()
print("=" * 62)
print("ROBUSTNESS: weather trend rates across window definitions")
print("  (per decade so windows of different lengths are comparable)")
print("=" * 62)
print(robust.round(2).to_string())


# ── optional: full projection under each window ────────────────────────────────
def projection_for_window(y0: int, y1: int) -> dict:
    """Re-estimate trends from window [y0,y1]; sensitivities frozen at 2020-2024."""
    d_g = trend_per_decade(gdd_anom, y0, y1) / 10 * HORIZON
    d_k = trend_per_decade(kdd_anom, y0, y1) / 10 * HORIZON
    b_g, b_k = five_yr_end(gdd_path), five_yr_end(kdd_path)
    net = b_g * d_g + b_k * d_k
    return {
        "dGDD (30y equiv)": round(d_g, 3),
        "dKDD (30y equiv)": round(d_k, 3),
        "net_logpts":       round(to_logpoints(net), 3),
        "net_pct":          round(to_pct(net), 2),
    }

proj_robust = pd.DataFrame(
    {label: projection_for_window(y0, y1) for label, y0, y1 in WINDOWS}
).T

print()
print("Implied net projection under each trend window")
print("  (sensitivities frozen at 2020-2024; CI on headline window only)")
print(proj_robust.to_string())
