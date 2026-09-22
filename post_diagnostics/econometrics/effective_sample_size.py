# -*- coding: utf-8 -*-
"""
Calculates and plots the effective sample size (N_eff) for an LLDVE model run.

This script does the following:
1.  Loads the dependent variable matrix (mY), the list of regions, and the
    optimal bandwidth (h) from a specified results folder.
2.  Aligns the mY data to ensure it matches the regions used in the model.
3.  Calculates N_eff(τ), the number of non-missing observations within the
    kernel window [τ - h, τ + h], for each time point τ.
4.  Plots N_eff(τ) against τ and warns if it falls below a threshold.
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# results_config.py lives in the repo root, two folders up from here
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import results_config as cfg

def run_ess_analysis():
    """Main function to calculate and plot effective sample size."""
    
    # ── which run to diagnose ──────────────────────────────────────────────
    # The paper's main run: corn, the four rainfed states pooled. The dataset
    # and the folder layout come from results_config.py in the repo root.
    final_res_subfolder = "17_18_19_27_model_3_GS_logY_anomalyX"
    results_dir = cfg.run_dir(final_res_subfolder)
    
    # --- Load Data ---
    my_path = os.path.join(results_dir, "input_mY.csv")
    regions_path = os.path.join(results_dir, "regions_list.csv")
    h_path = os.path.join(results_dir, "h_optimal.txt")
    
    required_files = [my_path, regions_path, h_path]
    if not all(os.path.exists(p) for p in required_files):
        print(f"Error: One or more required files not found in:\n{results_dir}")
        print("Please ensure 'input_mY.csv', 'regions_list.csv', and 'h_optimal.txt' exist.")
        sys.exit(1)
        
    mY_full = pd.read_csv(my_path, index_col=0)
    df_regions = pd.read_csv(regions_path)
    with open(h_path, 'r') as f:
        h_optimal = float(f.read().strip())
        
    print(f"Successfully loaded data from '{final_res_subfolder}'.\n")
    print(f"Using optimal bandwidth h = {h_optimal:.4f}")

    # --- Align Data ---
    try:
        if 'region_name' in df_regions.columns:
            model_counties = df_regions['region_name'].tolist() 
        else:
            model_counties = df_regions.iloc[:, 0].tolist()
    except Exception as e:
        print(f"Error reading regions list: {e}")
        sys.exit(1)

    # Filter mY to only include counties used in the model
    mY = mY_full[model_counties]
    
    # --- Calculate Effective Sample Size (N_eff) ---
    T, N = mY.shape
    taus = np.linspace(0, 1, T)
    
    N_eff = np.zeros(T, dtype=int)
    
    # Define the normalized time vector τ on the [0, 1] interval
    t_frac = np.arange(T) / (T - 1)

    for ell, tau in enumerate(taus):
        # Create a boolean mask for observations within the kernel window
        in_window = np.abs(t_frac - tau) <= h_optimal
        
        # Select the sub-matrix of observations within the window
        sub_mY = mY.iloc[in_window, :]
        
        # Count the total number of non-missing observations in this sub-matrix
        N_eff[ell] = sub_mY.notna().sum().sum()

    # --- Plot N_eff(τ) ---
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 5))
    plt.plot(taus, N_eff, marker='.', linestyle='-', markersize=8, label=r'$N_{eff}(\tau)$')
    
    # Plot formatting
    plt.xlabel(r"Normalized Time ($\tau$)", fontsize=12)
    plt.ylabel(r"Effective Sample Size ($N_{eff}$)", fontsize=12)
    plt.title(f"Effective Sample Size vs. Estimation Point (h = {h_optimal:.3f})", fontsize=14)
    plt.xlim(0, 1)
    plt.ylim(bottom=0)
    
    threshold = 30
    plt.axhline(threshold, color='red', linestyle='--', linewidth=1.5, label=f"Threshold = {threshold}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Identify and Report Issues ---
    low_idxs = np.where(N_eff < threshold)[0]
    if len(low_idxs) > 0:
        print(f"\nWarning: Effective sample size is below the threshold of {threshold} at the following points:")
        for idx in low_idxs:
            # Only print for boundary regions where this is most common
            if taus[idx] < h_optimal or taus[idx] > 1 - h_optimal:
                print(f"  τ = {taus[idx]:.3f}  →  N_eff = {N_eff[idx]}")
        print("\nThis is expected at the boundaries (τ < h and τ > 1-h) due to fewer observations in the window.")
        print("Consider trimming the estimation range to [h, 1-h] when interpreting results.")
    else:
        print(f"\nAnalysis complete: Effective sample size is above the threshold of {threshold} for all τ.")


if __name__ == '__main__':
    run_ess_analysis()