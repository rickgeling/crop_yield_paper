# -*- coding: utf-8 -*-
"""
Performs cross-sectional dependence diagnostics on LLDVE model residuals.

This script does the following:
1.  Loads and aligns the panel of residuals using the definitive regions list.
2.  Performs Pesaran's CD test for cross-sectional dependence.
3.  Visualizes the county-to-county residual correlation matrix with a heatmap.
4.  Calculates the condition number of the cross-section covariance matrix to
    check for near-singularity.
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# results_config.py lives in the repo root, two folders up from here
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import results_config as cfg
from scipy.stats import norm

def run_csd_diagnostics():
    """Main function to run all diagnostic tests and visualizations."""
    
    # ── which run to diagnose ──────────────────────────────────────────────
    # The paper's main run: corn, the four rainfed states pooled. The dataset
    # and the folder layout come from results_config.py in the repo root.
    final_res_subfolder = "17_18_19_27_model_3_GS_logY_anomalyX"
    results_dir = cfg.run_dir(final_res_subfolder)
    
    # --- Load and Align Data ---
    resid_path = os.path.join(results_dir, "residuals_final.csv")
    regions_path = os.path.join(results_dir, "regions_list.csv")
    
    required_files = [resid_path, regions_path]
    if not all(os.path.exists(p) for p in required_files):
        print(f"Error: One or more required files not found in:\n{results_dir}")
        print("Please ensure 'residuals_final.csv' and 'regions_list.csv' exist.")
        sys.exit(1)
        
    residuals_full = pd.read_csv(resid_path, index_col=0)
    df_regions = pd.read_csv(regions_path)
    print(f"Successfully loaded data from '{final_res_subfolder}'.\n")

    try:
        if 'region_name' in df_regions.columns:
            model_counties = df_regions['region_name'].tolist() 
        else:
            model_counties = df_regions.iloc[:, 0].tolist()
    except Exception as e:
        print(f"Error reading regions list: {e}")
        sys.exit(1)

    residuals = residuals_full[model_counties]
    T, N = residuals.shape
    print(f"Performing diagnostics on a panel of T={T} years and N={N} counties.")

    # --- 1. Pesaran’s CD Test (Entire Panel) ---
    print("\n--- Pesaran’s CD Test ---")
    # We compute pairwise correlations of ε̂_it across counties (i)
    # and form the CD statistic:
    #
    #   CD = sqrt(2 / [N(N-1)]) * sum_{i<j} ρ̂_{ij} * sqrt(T)
    
    R = residuals.corr()
    
    if N > 1:
        iu = np.triu_indices(N, k=1)
        rho_vals = R.values[iu]
        
        cd_stat = np.sqrt(2.0 / (N*(N-1))) * np.sqrt(T) * rho_vals.sum()
        p_value_cd = 2 * (1 - norm.cdf(abs(cd_stat)))

        print(f"Null Hypothesis (H0): Residuals are cross-sectionally independent.")
        print(f"CD statistic = {cd_stat:.3f}")
        print(f"p-value      = {p_value_cd:.4f}")
        if p_value_cd < 0.05:
            print("Result: Reject H0. Evidence of significant cross-sectional dependence.")
        else:
            print("Result: Fail to reject H0. No evidence of cross-sectional dependence.")
    else:
        print("Skipping CD Test: Only one county in the panel.")
    print("-" * 55 + "\n")

    # --- 2. Heatmap of Pairwise Correlations ---
    print("--- 2. Heatmap of Residual Correlations ---")
    plt.figure(figsize=(9, 8))
    plt.imshow(R, vmin=-1, vmax=1, cmap="RdBu_r")
    plt.colorbar(label="Residual Correlation ρ̂_ij")
    plt.title("Heatmap of County-to-County Residual Correlations")
    
    # To avoid unreadable labels, only show ticks for a subset of counties if N is large
    tick_step = 1 if N <= 50 else int(N / 25)
    plt.xticks(np.arange(0, N, tick_step), R.columns[::tick_step], rotation=90, fontsize=8)
    plt.yticks(np.arange(0, N, tick_step), R.index[::tick_step], fontsize=8)
    
    plt.tight_layout()
    plt.show()
    print("Generated heatmap of the residual correlation matrix.")
    print("-" * 55 + "\n")

    # --- 3. Condition Number of Cross-Sectional Covariance Matrix ---
    
    # Σ̂_cross = (1/T) ∑_t (ε̂_t - μ)(ε̂_t - μ)' where ε̂_t is the N×1 vector at time t,

    # and μ = T⁻¹ ∑_t ε̂_t is the time‐mean across t.
    
    
    print("--- 3. Cross-Section Covariance Condition Number ---")
    eps = residuals.values
    mu  = eps.mean(axis=0, keepdims=True)
    eps_centered = eps - mu
    
    Sigma_cross = (eps_centered.T @ eps_centered) / T
    
    try:
        eigvals = np.linalg.eigvalsh(Sigma_cross)
        print(eigvals)
        # Avoid division by zero if the smallest eigenvalue is ~0
        if eigvals[0] < 1e-10:
             cond_number = np.inf
        else:
            cond_number = eigvals[-1] / eigvals[0]

        print(f"Condition number κ = {cond_number:.2e}")

        if cond_number > 1e6:
            print("Warning: Σ̂_cross is ill-conditioned (nearly singular).")
            print("This can be an issue for some bootstrap methods but is often handled by the AWB.")
        else:
            print("Σ̂_cross is well-conditioned.")
    except Exception as e:
        print(f"Could not compute condition number. Error: {e}")

if __name__ == '__main__':
    run_csd_diagnostics()