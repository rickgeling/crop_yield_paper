import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Paths & Configuration ─────────────────────────────────────────────────────
# ── paths (dataset and run are set in results_config.py at the repo root) ─────
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import results_config as cfg

BASE_DIR = cfg.run_dir()
CI_DIR   = cfg.ci_dir()
RUN_NOTE = cfg.RUN_TAG

# Map the subplot titles to their corresponding file prefixes
VARIABLES = [
    ("Global Trend", "GT"),
    ("GDD", "GDDGS_anom"),
    ("KDD", "KDDGS_anom"),
    ("Precip", "PrecGS_anom"),
    ("Total Precip²", "PrecSqGS_anom"),
    ("KDDxPrec", "KDDxPrecGS_anom")
]

YEARS = np.arange(1951, 2025)

# ── Styling ───────────────────────────────────────────────────────────────────
def set_nature_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'pdf.fonttype': 42,
        'axes.labelsize': 11,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.4,
        'grid.color': '#CCCCCC'
    })

# ── Data Loading ──────────────────────────────────────────────────────────────
def load_ci_data(prefix):
    file_name = f"{prefix}_ST_PW_{RUN_NOTE}.csv"
    file_path = os.path.join(CI_DIR, file_name)
    if not os.path.exists(file_path):
        return None
    return pd.read_csv(file_path)

# ── Plot Generation ───────────────────────────────────────────────────────────
def generate_nature_plot():
    set_nature_style()
    
    # Balanced aspect ratio for the 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
    fig.suptitle('Time-Varying Coefficients: Pooled State Analysis (Growing Season)', fontsize=15, y=0.98)

    # Flatten the 2D axes array to iterate over it easily in 1D
    axes_flat = axes.flatten()

    # Nature-friendly colorblind palette
    color_main = '#D55E00'  # Vermilion for the main estimate
    color_fill = '#E69F00'  # Orange for the pointwise CI
    color_sim = '#0072B2'   # Blue for the simultaneous CI

    for idx, (title, prefix) in enumerate(VARIABLES):
        ax = axes_flat[idx]
        df = load_ci_data(prefix)

        if df is None:
            ax.text(0.5, 0.5, 'Data Unavailable', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            continue

        # Extract data vectors
        beta_hat = df['Beta_hat'].values
        pt_lb = df['Pointwise_LB'].values
        pt_ub = df['Pointwise_UB'].values
        sim_lb = df['Simult_LB'].values
        sim_ub = df['Simult_UB'].values

        # Plot components
        ax.plot(YEARS, beta_hat, color=color_main, lw=2.0, label='LLDVE Estimate')
        ax.fill_between(YEARS, pt_lb, pt_ub, color=color_fill, alpha=0.35, label='95% Pointwise CI')
        ax.plot(YEARS, sim_lb, color=color_sim, linestyle='--', lw=1.2, label='95% Simultaneous CI')
        ax.plot(YEARS, sim_ub, color=color_sim, linestyle='--', lw=1.2)

        # Baseline and formatting
        ax.axhline(0, color='black', lw=0.8, alpha=0.7)
        ax.set_title(title, pad=10)
        
        # Axis labeling: Y-axis only on the left column, X-axis only on the bottom row
        if idx % 3 == 0:
            ax.set_ylabel('Coefficient\n(GS)')
        
        if idx >= 3:
            ax.set_xlabel('Year')
            
        ax.set_xlim([1951, 2024])
        
        # ADJUSTMENT: Ticks every 10 years
        ax.xaxis.set_major_locator(plt.MultipleLocator(10))
        
        # Optimize tick display
        ax.tick_params(axis='both', length=0)
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5, prune='both'))

    # Adjust layout to make room for legend and title
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    
    # Construct unified legend at the very bottom
    handles, labels = axes_flat[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.01), frameon=False)

    # Export output
    output_path = os.path.join(BASE_DIR, "nature_style_lldve_gs_plot_final.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Plot successfully saved to: {output_path}")

if __name__ == "__main__":
    generate_nature_plot()