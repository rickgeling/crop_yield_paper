import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats

# ── paths ──────────────────────────────────────────────────────────────────────
# ── paths (dataset and run are set in results_config.py at the repo root) ─────
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import results_config as cfg

BASE    = cfg.run_dir()
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

YEARS   = np.arange(1951, 2025)
Y0, Y1  = 1994, 2024          # trend window used in impact projections

# ── load and extract ───────────────────────────────────────────────────────────
mX  = np.load(os.path.join(BASE, "input_mX.npy"))   # (N=334, T=74, d=5)
gdd = mX[:, :, 0]   # (N, T)
kdd = mX[:, :, 1]   # (N, T)

# ── cross-county summary statistics ───────────────────────────────────────────
def summary(arr):
    return {
        "mean":  arr.mean(axis=0),
        "p10":   np.percentile(arr, 10, axis=0),
        "p25":   np.percentile(arr, 25, axis=0),
        "p75":   np.percentile(arr, 75, axis=0),
        "p90":   np.percentile(arr, 90, axis=0),
    }

gs = summary(gdd)
ks = summary(kdd)

# ── linear trend over the projection window ────────────────────────────────────
def fit_trend_line(mean_series, y0, y1):
    mask  = (YEARS >= y0) & (YEARS <= y1)
    x     = YEARS[mask].astype(float)
    y     = mean_series[mask]
    slope, intercept, *_ = stats.linregress(x, y)
    x_ext = np.array([y0, y1], dtype=float)
    return x_ext, slope * x_ext + intercept, slope

gdd_tx, gdd_ty, gdd_slope = fit_trend_line(gs["mean"], Y0, Y1)
kdd_tx, kdd_ty, kdd_slope = fit_trend_line(ks["mean"], Y0, Y1)

# ── plot ───────────────────────────────────────────────────────────────────────
C_GDD  = "#2e7d32"   # dark green  (beneficial)
C_KDD  = "#c62828"   # dark red    (damaging)
C_ZERO = "#9e9e9e"   # mid-grey

fig, axes = plt.subplots(
    2, 1, figsize=(10, 6.5), sharex=True,
    layout="constrained",
    gridspec_kw={"hspace": 0.08}
)

for ax, s, color, label, unit in [
    (axes[0], gs, C_GDD, "GDD anomaly (growing degree days)", "GDD"),
    (axes[1], ks, C_KDD, "KDD anomaly (killing degree days)", "KDD"),
]:
    # 10-90 outer band
    ax.fill_between(YEARS, s["p10"], s["p90"],
                    color=color, alpha=0.10, lw=0, label="10th-90th pct")
    # IQR inner band
    ax.fill_between(YEARS, s["p25"], s["p75"],
                    color=color, alpha=0.22, lw=0, label="IQR (25th-75th)")
    # cross-county mean
    ax.plot(YEARS, s["mean"], color=color, lw=1.8, label="County mean", zorder=3)

    # zero baseline
    ax.axhline(0, color=C_ZERO, lw=0.8, ls="--", zorder=1)

    # trend window marker
    ax.axvline(Y0, color=C_ZERO, lw=0.7, ls=":", zorder=1)

    # linear trend line over [Y0, Y1]
    tx, ty = (gdd_tx, gdd_ty) if unit == "GDD" else (kdd_tx, kdd_ty)
    sl     = gdd_slope          if unit == "GDD" else kdd_slope
    ax.plot(tx, ty, color=color, lw=2.2, ls="--", zorder=4,
            label=f"1994-2024 trend ({sl*10:+.1f}/decade)")

    ax.set_ylabel(label, fontsize=10)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(6, integer=False))
    ax.tick_params(labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.6)

axes[1].set_xlabel("Year", fontsize=10)
axes[1].set_xlim(YEARS[0], YEARS[-1])
axes[1].xaxis.set_major_locator(mticker.MultipleLocator(10))

fig.suptitle(
    "Growing-season weather anomalies  (1981-2010 baseline)\n"
    "Illinois, Indiana, Iowa, Minnesota  |  county-level, 1951-2024",
    fontsize=11, y=1.01
)

out_path = os.path.join(OUT_DIR, "GDD_KDD_full_sample.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.show()
