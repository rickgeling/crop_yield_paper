import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── 1. Configuration & Paths ──────────────────────────────────────────────────
# ── paths (dataset and run are set in results_config.py at the repo root) ─────
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import results_config as cfg

BASE_RESULTS_PATH = cfg.run_dir()
MASTER_CSV_PATH   = cfg.master_panel_path()

# States to include for the pooled climatology calculation (Iowa, Illinois, Indiana, Minnesota)
STATES_FOR_CLIM = ["17", "18", "19", "27"]

# ── 2. Nature-Style Aesthetics ────────────────────────────────────────────────
def set_nature_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'pdf.fonttype': 42,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.4,
        'grid.color': '#CCCCCC'
    })

# ── 3. Data Loading Helpers ───────────────────────────────────────────────────
def load_lldve_coefficients(base_path):
    """Loads the main Theta estimates for the target run."""
    theta_path = os.path.join(base_path, "mTheta_hat.csv")
    if not os.path.exists(theta_path):
        raise FileNotFoundError(f"Missing coefficient file at {theta_path}")
        
    df = pd.read_csv(theta_path, index_col=0)
    
    # Heuristic to recover calendar years from the scaled index
    vals = df.index.astype(float).values
    if (vals.min() >= 0.0) and (vals.max() <= 10.0):
        years = np.arange(1951, 1951 + len(vals), dtype=int)
    else:
        years = np.rint(vals).astype(int)
        
    df['Year'] = years
    df.set_index('Year', inplace=True)
    
    # Restrict the data to 1951 through 2023 (skipping 2024)
    df = df[df.index <= 2023]
    return df

def compute_pooled_tref(csv_path, states):
    """Computes the baseline reference temperature from the master dataset."""
    df = pd.read_csv(csv_path, dtype={'state': str})
    df["state"] = df["state"].str.zfill(2)
    
    # Locate the TMAX column
    candidates = ["TMAX_AVG_GS", "tmax_avg_gs", "TMAX_GS_AVG", "tmax_gs_avg"]
    tmax_col = next((c for c in candidates if c in df.columns), None)
    
    if not tmax_col:
        print("Warning: TMAX column not found. Defaulting T_ref to 22.0 °C")
        return 22.0
        
    df_sub = df[(df["year"] >= 1951) & (df["year"] <= 2023)]
    df_sub = df_sub[df_sub["state"].isin(states)]
    df_sub = df_sub[np.isfinite(df_sub[tmax_col])]
    
    return float(df_sub[tmax_col].mean())

# ── 4. Math & Response Curve Functions ────────────────────────────────────────
def degree_days_from_t(t_grid, t_base=10.0, t_thr=29.0):
    """Maps a vector of temperatures to beneficial (GDD) and extreme (KDD) heat."""
    t_grid = np.asarray(t_grid, dtype=float)
    g = np.maximum(0.0, np.minimum(t_grid, t_thr) - t_base)
    k = np.maximum(0.0, t_grid - t_thr)
    return g, k

def calculate_yield_response(bar_gdd, bar_kdd, t_grid, t_ref):
    """Calculates the recentered log-yield response curve for a temperature grid."""
    g, k = degree_days_from_t(t_grid)
    dlog = bar_gdd * g + bar_kdd * k
    
    # Recenter relative to reference temperature
    g_ref, k_ref = degree_days_from_t(np.array([t_ref]))
    dlog_ref = (bar_gdd * g_ref + bar_kdd * k_ref)[0]
    
    return dlog - dlog_ref

# ── 5. Main Plotting Routine ──────────────────────────────────────────────────
def generate_3x3_temperature_grid():
    set_nature_style()
    
    # 1. Load Data
    print("Loading coefficient data...")
    df_coef = load_lldve_coefficients(BASE_RESULTS_PATH)
    
    print("Computing baseline climatology...")
    t_ref = compute_pooled_tref(MASTER_CSV_PATH, STATES_FOR_CLIM)
    print(f"Computed Pooled T_ref: {t_ref:.2f} °C")
    
    # Identify GDD and KDD columns dynamically (or fallback to indices 0 and 1)
    gdd_col = next((c for c in df_coef.columns if 'GDD' in c.upper()), df_coef.columns[0])
    kdd_col = next((c for c in df_coef.columns if 'KDD' in c.upper()), df_coef.columns[1])
    
    # 2. Define Time Windows (8 windows of 8 years, 1 window of 9 years)
    windows = [(1951 + i*8, 1951 + i*8 + 7) for i in range(8)]
    windows.append((2015, 2023)) 
    
    # 3. Setup Plot
    t_grid = np.linspace(10.0, 40.0, 601)
    fig, axes = plt.subplots(3, 3, figsize=(11, 10), sharex=True, sharey=True)
    fig.suptitle('Evolving Temperature Response: Pooled Maize Yields (1951–2023)', 
                 fontsize=16, y=0.96)
                 
    axes_flat = axes.flatten()
    color_curve = '#D55E00'  # Nature-style Vermilion
    
    # 4. Iterate over grid and plot
    for idx, (y_start, y_end) in enumerate(windows):
        ax = axes_flat[idx]
        
        # Mask data for the current window
        mask = (df_coef.index >= y_start) & (df_coef.index <= y_end)
        df_window = df_coef[mask]
        
        if df_window.empty:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{y_start}–{y_end}")
            continue
            
        # Average the coefficients for the window
        mean_gdd = df_window[gdd_col].mean()
        mean_kdd = df_window[kdd_col].mean()
        
        # Calculate curve
        curve = calculate_yield_response(mean_gdd, mean_kdd, t_grid, t_ref)
        
        # Plot curve and reference lines
        ax.plot(t_grid, curve, color=color_curve, lw=2.5)
        ax.axhline(0, color='black', lw=1.0, alpha=0.7, linestyle=':')
        ax.axvline(29, color='#0072B2', lw=1.2, linestyle='--', alpha=0.7) # 29°C threshold
        
        # Formatting
        ax.set_title(f"{y_start}–{y_end} (Avg)", pad=8)
        
        # Tighter X-axis bounds
        ax.set_xlim([15, 38])
        ax.xaxis.set_major_locator(plt.MultipleLocator(5))
        
        # ADJUSTMENT: Much tighter Y-axis bounds to reveal the curve's shape
        ax.set_ylim([-0.06, 0.02]) 
        
        # Axis labeling: Y-axis on left column, X-axis on bottom row
        if idx % 3 == 0:
            ax.set_ylabel(f'Δ log Yield\n(relative to {t_ref:.1f}°C)')
        
        if idx >= 6:
            ax.set_xlabel('Daily Maximum Temperature (°C)')

    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    
    # Export output
    output_path = os.path.join(BASE_RESULTS_PATH, "nature_style_temp_yield_3x3_grid.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Plot successfully saved to: {output_path}")

if __name__ == "__main__":
    generate_3x3_temperature_grid()