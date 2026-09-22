import os
import numpy as np
import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────
# ── paths (dataset and run are set in results_config.py at the repo root) ─────
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import results_config as cfg

BASE      = cfg.run_dir()
CI_DIR    = cfg.ci_dir()
BOOT_DIR  = os.path.join(BASE, "bootstrap_paths")   # optional: AWB replicate paths
MODEL_TAG = cfg.RUN_TAG

YEARS        = np.arange(1951, 2025)   # 74 years, rows 0-73
WINDOW       = 5                        # rolling window for peak detection
PREC_DEFICIT = 100                      # mm representative drought deficit

# representative weather shocks for the bu/acre translation.
# set these to ~1 SD of the respective GS anomaly so the shock is realistic.
SHOCK_GDD = 100    # +GDD anomaly (degree-days)
SHOCK_KDD = 30     # +KDD anomaly (degree-days)

ALPHA = 0.05       # for bootstrap CIs (95%)


# ── helpers ───────────────────────────────────────────────────────────────────

def load_coef(var: str) -> pd.Series:
    fname = f"{var}_ST_PW_{MODEL_TAG}.csv"
    df    = pd.read_csv(os.path.join(CI_DIR, fname))
    return pd.Series(df["Beta_hat"].values, index=YEARS)


def load_boot_paths(var: str):
    """
    Optional. Returns an (n_boot, 74) array of AWB replicate coefficient paths,
    or None if the file is not present. Expected layout: one replicate per row,
    one column per year. Adjust the filename/loader to match how you saved them.
    """
    npy = os.path.join(BOOT_DIR, f"{var}_BOOT_{MODEL_TAG}.npy")
    csv = os.path.join(BOOT_DIR, f"{var}_BOOT_{MODEL_TAG}.csv")
    if os.path.exists(npy):
        return np.load(npy)
    if os.path.exists(csv):
        return pd.read_csv(csv).values
    return None


def rolling_window_peak(series: pd.Series, mode: str):
    """
    Returns (start_year, end_year, avg_value) for the WINDOW-year rolling window
    whose average is highest (mode='max') or lowest/most-negative (mode='min').
    Rolling index label = last year in the window.
    """
    roll     = series.rolling(WINDOW).mean().dropna()
    peak_idx = roll.idxmax() if mode == "max" else roll.idxmin()
    return peak_idx - WINDOW + 1, peak_idx, roll[peak_idx]


def instantaneous_peak(series: pd.Series, mode: str):
    """Single best year (argmax / argmin) and its coefficient value."""
    idx = series.idxmax() if mode == "max" else series.idxmin()
    return idx, series[idx]


def avg_window(series: pd.Series, start: int, end: int) -> float:
    return series.loc[start:end].mean()


def first_5yr(series: pd.Series) -> float:
    return series.iloc[:WINDOW].mean()


def last_5yr(series: pd.Series) -> float:
    return series.iloc[-WINDOW:].mean()


def yield_impact_bu(beta: float, shock: float, log_baseline: float) -> float:
    """bu/acre change from a `shock`-unit anomaly at a baseline log-yield level."""
    return np.exp(log_baseline) * (np.exp(beta * shock) - 1)


def boot_ci(boot_paths, func, alpha: float = ALPHA):
    """
    boot_paths : (n_boot, 74) AWB replicate coefficient paths, or None.
    func       : maps a coefficient path (pd.Series indexed by YEARS) -> scalar.
    Returns (lo, hi) or (None, None) if no draws are available.
    """
    if boot_paths is None:
        return None, None
    stats = np.array([func(pd.Series(p, index=YEARS)) for p in boot_paths])
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi


def boot_ci_joint(paths_a, paths_b, func, alpha: float = ALPHA):
    """
    For statistics that mix two coefficients (e.g. the amplification ratio),
    propagate both draws together within each replicate to preserve their
    joint variation. Replicates are matched by row index, so the two arrays
    must share the same bootstrap ordering.
    """
    if paths_a is None or paths_b is None:
        return None, None
    n = min(len(paths_a), len(paths_b))
    stats = np.array([
        func(pd.Series(paths_a[i], index=YEARS),
             pd.Series(paths_b[i], index=YEARS))
        for i in range(n)
    ])
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi


def fmt(v, dec=5):
    return f"{v:.{dec}f}"


def pct(v, dec=1):
    return f"{v:.{dec}f}%"


def ci_str(lo, hi, dec=5):
    if lo is None:
        return "[CI unavailable: no bootstrap paths found]"
    return f"[{lo:.{dec}f}, {hi:.{dec}f}]"


# ── load coefficients ─────────────────────────────────────────────────────────
gt    = load_coef("GT")
gdd   = load_coef("GDDGS_anom")
kdd   = load_coef("KDDGS_anom")
inter = load_coef("KDDxPrecGS_anom")

# optional AWB replicate paths (None if not saved)
gt_b    = load_boot_paths("GT")
gdd_b   = load_boot_paths("GDDGS_anom")
kdd_b   = load_boot_paths("KDDGS_anom")
inter_b = load_boot_paths("KDDxPrecGS_anom")

# current normal-weather log-yield baseline (last 5 yrs of the global trend)
log_base_now = last_5yr(gt)


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  GLOBAL TREND
# ═══════════════════════════════════════════════════════════════════════════════
gt_1951 = gt.iloc[0]
gt_2024 = gt.iloc[-1]
gt_pct  = (np.exp(gt_2024 - gt_1951) - 1) * 100   # log-level Δ → % yield change

print("=" * 62)
print("GLOBAL TREND  (log-yield units)")
print("=" * 62)
print(f"  1951 (year 1):  {fmt(gt_1951, 4)}  (≈ {np.exp(gt_1951):.0f} bu/acre)")
print(f"  2024 (year 74): {fmt(gt_2024, 4)}  (≈ {np.exp(gt_2024):.0f} bu/acre)")
print(f"  Implied cumulative yield change: {gt_pct:+.1f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  GDD  (beneficial heat – positive, peaked ~1990s, then declined)
# ═══════════════════════════════════════════════════════════════════════════════
gdd_first5               = first_5yr(gdd)
gdd_last5                = last_5yr(gdd)
gdd_ps, gdd_pe, gdd_peak = rolling_window_peak(gdd, mode="max")
gdd_pk_yr, gdd_pk_val    = instantaneous_peak(gdd, mode="max")

# % decline from peak: (peak − last5) / |peak| × 100
gdd_pct_decline = (gdd_peak - gdd_last5) / abs(gdd_peak) * 100

# bu/acre translation at a FIXED current baseline (isolates the coefficient move)
gdd_benefit_peak = yield_impact_bu(gdd_peak,  SHOCK_GDD, log_base_now)
gdd_benefit_end  = yield_impact_bu(gdd_last5, SHOCK_GDD, log_base_now)
gdd_benefit_lost = gdd_benefit_peak - gdd_benefit_end

gdd_benefit_peak_1 = yield_impact_bu(gdd_peak,  1, log_base_now)
gdd_benefit_end_1  = yield_impact_bu(gdd_last5, 1, log_base_now)
gdd_benefit_lost_1 = gdd_benefit_peak_1 - gdd_benefit_end_1

# bootstrap CIs on the averaged quantities (fixed peak window → no winner's-curse bias)
ci_gdd_last5 = boot_ci(gdd_b, last_5yr)
ci_gdd_peak  = boot_ci(gdd_b, lambda s: avg_window(s, gdd_ps, gdd_pe))

print("\n" + "=" * 62)
print("GDD  (beneficial temperature – growing degree days)")
print("=" * 62)
print(f"  First-5-yr avg  1951–1955:      {fmt(gdd_first5)}")
print(f"  Single peak year:               {gdd_pk_yr}  ({fmt(gdd_pk_val)})")
print(f"  Peak 5-yr avg   {gdd_ps}–{gdd_pe}:    {fmt(gdd_peak)}  ← highest  CI {ci_str(*ci_gdd_peak)}")
print(f"  Last-5-yr avg   2020–2024:      {fmt(gdd_last5)}  CI {ci_str(*ci_gdd_last5)}")
print(f"  % declined  peak → end:         {pct(gdd_pct_decline)}")
print(f"  bu/acre @ +{SHOCK_GDD} GDD shock (fixed {np.exp(log_base_now):.0f} bu/acre baseline):")
print(f"    benefit at peak:   {gdd_benefit_peak:+.2f} bu/acre")
print(f"    benefit at end:    {gdd_benefit_end:+.2f} bu/acre")
print(f"    benefit lost:      {gdd_benefit_lost:.2f} bu/acre")
print(f"  bu/acre @ +1 GDD shock (fixed {np.exp(log_base_now):.0f} bu/acre baseline):")
print(f"    benefit at peak:   {gdd_benefit_peak_1:+.2f} bu/acre")
print(f"    benefit at end:    {gdd_benefit_end_1:+.2f} bu/acre")
print(f"    benefit lost:      {gdd_benefit_lost_1:.2f} bu/acre")


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  KDD  (extreme heat – negative, worsened to a peak, then partially recovered)
# ═══════════════════════════════════════════════════════════════════════════════
kdd_first5               = first_5yr(kdd)
kdd_last5                = last_5yr(kdd)
kdd_ps, kdd_pe, kdd_peak = rolling_window_peak(kdd, mode="min")  # most negative
kdd_pk_yr, kdd_pk_val    = instantaneous_peak(kdd, mode="min")

# % recovered: (last5 − peak) / |peak| × 100  → share of peak damage undone
kdd_pct_recovery = (kdd_last5 - kdd_peak) / abs(kdd_peak) * 100

# bu/acre translation at the same FIXED current baseline
kdd_damage_peak = yield_impact_bu(kdd_peak,  SHOCK_KDD, log_base_now)  # most harm
kdd_damage_end  = yield_impact_bu(kdd_last5, SHOCK_KDD, log_base_now)
kdd_recovered   = kdd_damage_end - kdd_damage_peak                     # +ve = less harm now

kdd_damage_peak_1 = yield_impact_bu(kdd_peak,  1, log_base_now)
kdd_damage_end_1  = yield_impact_bu(kdd_last5, 1, log_base_now)
kdd_recovered_1   = kdd_damage_end_1 - kdd_damage_peak_1

ci_kdd_last5 = boot_ci(kdd_b, last_5yr)
ci_kdd_peak  = boot_ci(kdd_b, lambda s: avg_window(s, kdd_ps, kdd_pe))

print("\n" + "=" * 62)
print("KDD  (extreme heat – killing degree days, negative coefs)")
print("=" * 62)
print(f"  First-5-yr avg  1951–1955:      {fmt(kdd_first5)}  ← least damage")
print(f"  Single worst year:              {kdd_pk_yr}  ({fmt(kdd_pk_val)})")
print(f"  Peak 5-yr avg   {kdd_ps}–{kdd_pe}:    {fmt(kdd_peak)}  ← most negative  CI {ci_str(*ci_kdd_peak)}")
print(f"  Last-5-yr avg   2020–2024:      {fmt(kdd_last5)}  CI {ci_str(*ci_kdd_last5)}")
print(f"  % recovered relative to peak:   {pct(kdd_pct_recovery)}")
print(f"  bu/acre @ +{SHOCK_KDD} KDD shock (fixed {np.exp(log_base_now):.0f} bu/acre baseline):")
print(f"    damage at peak:    {kdd_damage_peak:+.2f} bu/acre")
print(f"    damage at end:     {kdd_damage_end:+.2f} bu/acre")
print(f"    damage recovered:  {kdd_recovered:+.2f} bu/acre")
print(f"  bu/acre @ +1 KDD shock (fixed {np.exp(log_base_now):.0f} bu/acre baseline):")
print(f"    damage at peak:    {kdd_damage_peak_1:+.2f} bu/acre")
print(f"    damage at end:     {kdd_damage_end_1:+.2f} bu/acre")
print(f"    damage recovered:  {kdd_recovered_1:+.2f} bu/acre")


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  KDD × PRECIPITATION  (interaction – positive; mitigates or amplifies)
# ═══════════════════════════════════════════════════════════════════════════════
inter_first5  = first_5yr(inter)
inter_last5   = last_5yr(inter)
inter_at_peak = avg_window(inter, kdd_ps, kdd_pe)   # interaction over the KDD-peak window

# Amplification % at a -PREC_DEFICIT mm deficit, relative to each era's own |β_KDD|:
#   extra KDD damage under drought = β_inter × PREC_DEFICIT
#   amp% = (β_inter × PREC_DEFICIT) / |β_KDD| × 100
amp_peak = (inter_at_peak * PREC_DEFICIT) / abs(kdd_peak)  * 100
amp_end  = (inter_last5   * PREC_DEFICIT) / abs(kdd_last5) * 100

# joint bootstrap CI on the end-of-sample amplification (mixes inter and kdd draws)
def _amp_end_stat(inter_path, kdd_path):
    return (last_5yr(inter_path) * PREC_DEFICIT) / abs(last_5yr(kdd_path)) * 100
ci_amp_end = boot_ci_joint(inter_b, kdd_b, _amp_end_stat)

print("\n" + "=" * 62)
print("KDD × PRECIPITATION  (compound interaction)")
print("=" * 62)
print(f"  First-5-yr avg  1951–1955:        {inter_first5:.6f}")
print(f"  Last-5-yr avg   2020–2024:        {inter_last5:.6f}")
print(f"\n  Amplification at -{PREC_DEFICIT}mm precipitation deficit")
print(f"  (relative to each era's own |β_KDD|):")
print(f"    KDD-peak window {kdd_ps}–{kdd_pe}:  {pct(amp_peak)}")
print(f"    End of sample 2020–2024:    {pct(amp_end)}  CI {ci_str(ci_amp_end[0], ci_amp_end[1], dec=1)}")
print(f"    Change peak → end:          {amp_end - amp_peak:+.1f} percentage points")