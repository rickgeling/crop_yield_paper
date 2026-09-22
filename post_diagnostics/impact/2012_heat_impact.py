"""
2012 named-event vignette — implied yield hit, "then" vs "now" sensitivity.

A real 2012-magnitude event is run through the model at two epochs, with the
SHOCKS HELD FIXED at their observed 2012 values and only the sensitivity
coefficients changing between epochs.  "then" = the KDD-damage peak window;
"now" = the 2020-2024 end-of-sample average.

Three coherent objects (do not mix them into a fourth):

  (1) KDD only            — the extreme-heat component of 2012.  HEADLINE.
                            Conditions on the heat, says nothing about drought.
  (2) full 2012 (5 terms) — GDD + KDD + Prec + PrecSq + KDDxPrec, each at its
                            observed 2012 anomaly.  A complete, internally
                            consistent net for a whole 2012-type year, incl.
                            the drought's direct and compound channels.
  (3) GDD only [diag]     — answers whether the hot year's capped GDD provides
                            any offsetting benefit.  Context only, NOT a net.

The gap between (2) and (1) is the quantified "2012 was also a drought year"
point: how much the drought + compound channels add on top of pure heat.

NOTE: object (2) leans on the precipitation / compound coefficient paths, which
are still provisional.  Object (1) does not, which is why it is the headline.

Naming for the write-up: "a heat shock the size of 2012's, evaluated at the
crop's late-1990s sensitivity vs today" — the event size is fixed, only the
response coefficient moves.
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

YEARS      = np.arange(1951, 2025)   # 74 years, matches T in saved arrays
EVENT_YEAR = 2012
ALPHA      = 0.05


# ── helpers ────────────────────────────────────────────────────────────────────

def load_coef(var: str) -> pd.Series:
    fname = f"{var}_ST_PW_{MODEL_TAG}.csv"
    df    = pd.read_csv(os.path.join(CI_DIR, fname))
    return pd.Series(df["Beta_hat"].values, index=YEARS)


def five_yr_end(path: pd.Series) -> float:
    """'now' sensitivity: mean of final 5 years (2020-2024)."""
    return path.iloc[-5:].mean()


def avg_window(path: pd.Series, start: int, end: int) -> float:
    return path.loc[start:end].mean()


def rolling_window_peak(series: pd.Series, mode: str, window: int = 5):
    """(start, end, value) for the 5-yr rolling window with the most negative
       (mode='min') or highest (mode='max') average. Index label = last year."""
    roll = series.rolling(window).mean().dropna()
    pk   = roll.idxmin() if mode == "min" else roll.idxmax()
    return pk - window + 1, pk, roll[pk]


def to_pct(dln: float) -> float:
    """Exact log-to-percent."""
    return (np.exp(dln) - 1) * 100


def to_logpoints(dln: float) -> float:
    """Approximate percent (log points)."""
    return dln * 100


def ci(arr: np.ndarray):
    return np.percentile(arr, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])


# ── term registry ───────────────────────────────────────────────────────────────
# name -> (coefficient CSV stem, draws_all coef column, mX_raw variable column)
#   draws_all coef order:  GT(0) GDD(1) KDD(2) Prec(3) PrecSq(4) KDDxPrec(5)
#   mX_raw variable order: GDD(0) KDD(1) Prec(2) PrecSq(3) KDDxPrec(4)
# CHECK these CSV stems against your filenames; GDD/KDD are already verified.
COEF_TAGS = {
    "GDD":      ("GDDGS_anom",      1, 0),
    "KDD":      ("KDDGS_anom",      2, 1),
    "Prec":     ("PrecGS_anom",     3, 2),
    "PrecSq":   ("PrecSqGS_anom",   4, 3),
    "KDDxPrec": ("KDDxPrecGS_anom", 5, 4),
}


# ══════════════════════════════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════════════════════════════

# coefficient point-estimate paths
paths = {name: load_coef(tag) for name, (tag, _, _) in COEF_TAGS.items()}

# bootstrap draws: (B, T, n_coefs)
draws_all = np.load(os.path.join(BASE, "mTheta_stars_draws.npy"))
assert draws_all.shape[1] == len(YEARS), \
    f"T mismatch: draws have {draws_all.shape[1]} years, YEARS has {len(YEARS)}"
B = draws_all.shape[0]
boots = {name: draws_all[:, :, dcol] for name, (_, dcol, _) in COEF_TAGS.items()}

# county-year anomalies: (N, T, d)
mX_raw = np.load(os.path.join(BASE, "input_mX.npy"))

# fixed 2012 shocks (pooled county mean for each term, matches pooled coef scale)
ev    = int(np.where(YEARS == EVENT_YEAR)[0][0])
shock = {name: np.nanmean(mX_raw[:, ev, mx_i])
         for name, (_, _, mx_i) in COEF_TAGS.items()}

# epoch windows
kdd_ps, kdd_pe, _ = rolling_window_peak(paths["KDD"], mode="min")   # "then"
# "now" handled by five_yr_end


# ── epoch evaluation ─────────────────────────────────────────────────────────────

def epoch_beta(path: pd.Series, epoch: str) -> float:
    return five_yr_end(path) if epoch == "now" else avg_window(path, kdd_ps, kdd_pe)


def epoch_beta_boot(bmat: np.ndarray, i: int, epoch: str) -> float:
    s = pd.Series(bmat[i], index=YEARS)
    return five_yr_end(s) if epoch == "now" else avg_window(s, kdd_ps, kdd_pe)


def implied_loss(terms, epoch: str) -> float:
    """Point-estimate implied log-yield change from `terms` at `epoch`."""
    return sum(epoch_beta(paths[t], epoch) * shock[t] for t in terms)


def implied_loss_draws(terms, epoch: str) -> np.ndarray:
    """Within-replicate draws (joint across terms preserves their correlation)."""
    return np.array([
        sum(epoch_beta_boot(boots[t], i, epoch) * shock[t] for t in terms)
        for i in range(B)
    ])


# ══════════════════════════════════════════════════════════════════════════════
# THE THREE OBJECTS
# ══════════════════════════════════════════════════════════════════════════════
OBJECTS = {
    "(1) KDD only":            ["KDD"],
    "(2) full 2012 (5 terms)": ["GDD", "KDD", "Prec", "PrecSq", "KDDxPrec"],
    "(3) GDD only [diag]":     ["GDD"],
}

print("=" * 76)
print("2012 VIGNETTE — implied log-yield hit, then vs now (shocks fixed at 2012)")
print("=" * 76)
print("Observed 2012 pooled anomalies (units as stored in mX):")
for name in ["GDD", "KDD", "Prec", "PrecSq", "KDDxPrec"]:
    print(f"  {name:9s}: {shock[name]:+.3f}")
print(f"\n'then' = KDD peak window {kdd_ps}-{kdd_pe};  'now' = 2020-2024")
print(f"bootstrap replicates B = {B}\n")

rows = []
for label, terms in OBJECTS.items():
    pt_then = implied_loss(terms, "then")
    pt_now  = implied_loss(terms, "now")
    dr_then = implied_loss_draws(terms, "then")
    dr_now  = implied_loss_draws(terms, "now")
    dr_red  = dr_now - dr_then                      # >0 => smaller loss now

    tl, th = ci(dr_then)
    nl, nh = ci(dr_now)
    rl, rh = ci(dr_red)

    rows.append({
        "object":        label,
        "then":          to_logpoints(pt_then),
        "then_CI":       f"[{to_logpoints(tl):+.1f}, {to_logpoints(th):+.1f}]",
        "now":           to_logpoints(pt_now),
        "now_CI":        f"[{to_logpoints(nl):+.1f}, {to_logpoints(nh):+.1f}]",
        "reduction":     to_logpoints(pt_now - pt_then),
        "reduction_CI":  f"[{to_logpoints(rl):+.1f}, {to_logpoints(rh):+.1f}]",
    })

out = pd.DataFrame(rows).set_index("object")
pd.set_option("display.width", 160)
print("All values in LOG POINTS (100 x dln). 'reduction' = now - then; "
      "positive means less loss now.")
print(out.to_string(float_format=lambda x: f"{x:+.2f}"))


# ── exact-% headline for object (1) ──────────────────────────────────────────────
h_then = implied_loss(["KDD"], "then")
h_now  = implied_loss(["KDD"], "now")
print()
print("Headline (object 1, exact %): a 2012-sized heat shock implies "
      f"{to_pct(h_now):+.2f}% now vs {to_pct(h_then):+.2f}% at the {kdd_ps}-{kdd_pe} peak.")


# ── compound-stress gap: full minus heat-only ────────────────────────────────────
gap_then = implied_loss(OBJECTS["(2) full 2012 (5 terms)"], "then") - h_then
gap_now  = implied_loss(OBJECTS["(2) full 2012 (5 terms)"], "now")  - h_now
print()
print("Drought + compound contribution BEYOND pure heat in 2012:")
print(f"  at peak sensitivity: {to_logpoints(gap_then):+.2f} log pts")
print(f"  at now  sensitivity: {to_logpoints(gap_now):+.2f} log pts")
print("(this gap is the quantified 'dry-year deepens the loss' point; it leans")
print(" on the provisional precipitation / compound coefficients)")