# -*- coding: utf-8 -*-
"""
Performs serial correlation diagnostics on the residuals from an LLDVE model run.

This script does the following:
1.  Loads the panel of residuals from a specified LLDVE results folder.
2.  Performs the Wooldridge test for first-order autocorrelation in the panel data.
3.  Computes and summarizes the lag-1 autocorrelation (ρ_i(1)) for each county.
4.  Suggests a coefficient (γ) for the Autoregressive Wild Bootstrap based on the median ρ_i(1).
5.  Generates plots:
    - A histogram of the distribution of county-level autocorrelations.
    - Sample ACF plots for representative counties.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# results_config.py lives in the repo root, two folders up from here
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import results_config as cfg
from statsmodels.tsa.stattools import acf
# Note: The Wooldridge test is typically done with exogenous variables.
# This script performs a common and practical variant by regressing residuals
# on their lags, which is a strong indicator of serial correlation.

def run_serial_corr_diagnostics():
    """Main function to run all diagnostic tests and visualizations."""
    
    # ── which run to diagnose ──────────────────────────────────────────────
    # The paper's main run: corn, the four rainfed states pooled. The dataset
    # and the folder layout come from results_config.py in the repo root.
    final_res_subfolder = "17_18_19_27_model_3_GS_logY_anomalyX"
    results_dir = cfg.run_dir(final_res_subfolder)
    
    # --- CORRECTED: Build and check file-paths ---
    resid_path = os.path.join(results_dir, "residuals_final.csv")
    regions_path = os.path.join(results_dir, "regions_list.csv")
    
    required_files = [resid_path, regions_path]
    if not all(os.path.exists(p) for p in required_files):
        print(f"Error: One or more required files not found in:\n{results_dir}")
        print("Please ensure 'residuals_final.csv' and 'regions_list.csv' exist.")
        sys.exit(1)
        
    # --- CORRECTED: Load and Align Data ---
    residuals_full = pd.read_csv(resid_path, index_col=0)
    df_regions = pd.read_csv(regions_path)
    print(f"Successfully loaded data from '{final_res_subfolder}'.\n")

    # Get the definitive list of county names from the saved regions file
    try:
        if 'region_name' in df_regions.columns:
            model_counties = df_regions['region_name'].tolist() 
        elif 'region' in df_regions.columns:
            model_counties = df_regions['region'].tolist()
        else:
            model_counties = df_regions.iloc[:, 0].tolist()
    except Exception as e:
        print(f"Error reading regions list: {e}")
        sys.exit(1)

    # Filter the residuals DataFrame to only include counties used in the model
    residuals = residuals_full[model_counties]
    
    # --- FORMAL PANEL TEST: WOOLDRIDGE TEST ---
    print("--- Wooldridge Test for First-Order Autocorrelation ---")
    
    res_lag = residuals.shift(1)
    
    # CORRECTED version
    df_test = pd.DataFrame({
        'res_t': residuals.stack(future_stack=True),
        'res_t_minus_1': res_lag.stack(future_stack=True)
    }).dropna()
    
    y = df_test['res_t']
    X = df_test[['res_t_minus_1']]
    X = pd.concat([pd.Series(1, index=X.index, name="const"), X], axis=1)
    
    from statsmodels.api import OLS
    model = OLS(y, X).fit()
    rho_hat = model.params['res_t_minus_1']
    se_rho_hat = model.bse['res_t_minus_1']
    
    z_stat = rho_hat / se_rho_hat
    p_value = model.pvalues['res_t_minus_1']

    print(f"Null Hypothesis (H0): No first-order serial correlation (ρ = 0).")
    print(f"Estimated ρ on lagged residuals: {rho_hat:.4f}")
    print(f"Test Statistic (z): {z_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("Result: Reject H0. Evidence of significant serial correlation found.")
    else:
        print("Result: Fail to reject H0. No evidence of significant serial correlation.")
    print("-" * 55 + "\n")


    # --- COUNTY-LEVEL AUTOCORRELATION ANALYSIS ---
    print("--- County-Level Lag-1 Autocorrelation (ρ_i(1)) Analysis ---")
    
    rho1_list = [residuals[county].dropna().autocorr(lag=1) for county in residuals.columns]
    rho1_series = pd.Series(rho1_list, index=residuals.columns, name="rho1")
    
    median_rho1 = rho1_series.median()
    q25_rho1 = rho1_series.quantile(0.25)
    q75_rho1 = rho1_series.quantile(0.75)
    
    print("Summary of ρ_i(1) across all counties:")
    print(f"  Median = {median_rho1:.3f}")
    print(f"  Mean   = {rho1_series.mean():.3f}")
    print(f"  IQR    = {q75_rho1 - q25_rho1:.3f}  (25th={q25_rho1:.3f}, 75th={q75_rho1:.3f})")

    gamma = median_rho1
    print(f"\nSuggested AWB-AR coefficient γ = median(ρ_i(1)) = {gamma:.3f}")
    
    threshold = 0.5
    high_pers = rho1_series[rho1_series > threshold]
    print(f"\n{len(high_pers)} counties found with ρ_i(1) > {threshold}:")
    if not high_pers.empty:
        print(high_pers.sort_values(ascending=False).to_string())
    
    plt.figure(figsize=(7, 5))
    plt.hist(rho1_series.dropna(), bins=20, edgecolor='black', alpha=0.7)
    plt.axvline(median_rho1, color='red', linestyle='--', label=f"Median = {median_rho1:.2f}")
    plt.title("Histogram of County-Level Lag-1 Autocorrelations")
    plt.xlabel("Autocorrelation Coefficient (ρ_i(1))")
    plt.ylabel("Number of Counties")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    names_25 = rho1_series.sub(q25_rho1).abs().idxmin()
    names_50 = rho1_series.sub(median_rho1).abs().idxmin()
    names_75 = rho1_series.sub(q75_rho1).abs().idxmin()
    chosen_counties = [names_25, names_50, names_75]

    plt.figure(figsize=(8, 6))
    for county_name in chosen_counties:
        resids_county = residuals[county_name].dropna().values
        max_lags = min(10, len(resids_county) // 2 - 1)
        if max_lags > 0:
            acf_vals = acf(resids_county, nlags=max_lags, fft=False)
            plt.stem(range(len(acf_vals)), acf_vals, label=f"{county_name} (ρ₁≈{rho1_series[county_name]:.2f})")

    plt.axhline(0, color='black', linewidth=0.7)
    plt.title("Sample ACFs for Three Representative Counties")
    plt.xlabel("Lag (ℓ)")
    plt.ylabel("Autocorrelation ρ_i(ℓ)")
    plt.xticks(np.arange(0, max_lags + 1, 2))
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    run_serial_corr_diagnostics()