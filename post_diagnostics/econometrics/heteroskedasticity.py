import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# results_config.py lives in the repo root, two folders up from here
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import results_config as cfg
import statsmodels.api as sm
import statsmodels.stats.api as sms

def run_heteroskedasticity_diagnostics():
    """Main function to run all diagnostic tests and visualizations."""
    
    # ── which run to diagnose ──────────────────────────────────────────────
    # The paper's main run: corn, the four rainfed states pooled. The dataset
    # and the folder layout come from results_config.py in the repo root.
    final_res_subfolder = "17_18_19_27_model_3_GS_logY_anomalyX"
    results_dir = cfg.run_dir(final_res_subfolder)
    
    # Build and check file-paths
    resid_path = os.path.join(results_dir, "residuals_final.csv")
    fitted_path = os.path.join(results_dir, "fitted_values_final.csv")
    x_path = os.path.join(results_dir, "input_mX.npy")
    regions_path = os.path.join(results_dir, "regions_list.csv")

    # --- Load Data ---
    required_files = [resid_path, fitted_path, x_path, regions_path]
    if not all(os.path.exists(p) for p in required_files):
        print(f"Error: One or more required files not found in:\n{results_dir}")
        print("Please check your 'final_res_subfolder' name and the project structure.")
        sys.exit(1)
        
    residuals_full = pd.read_csv(resid_path, index_col=0)
    fitted_full = pd.read_csv(fitted_path, index_col=0)
    mX = np.load(x_path)
    df_regions = pd.read_csv(regions_path)
    print(f"Successfully loaded all required data from '{final_res_subfolder}'.\n")

    # --- Alignment ---
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

    residuals = residuals_full[model_counties]
    fitted = fitted_full[model_counties]

    # --- FORMAL PANEL TEST: BREUSCH-PAGAN TEST ---
    print("--- Breusch-Pagan Test for Heteroskedasticity ---")
    
    # CORRECTED: Unpack shape as (Counties, Time, Variables)
    N, T, K = mX.shape
    print(f"Interpreted mX shape as N={N} counties, T={T} years, K={K} variables.")
    
    # Reshape data for the test function.
    residuals_stacked = residuals.stack(dropna=False).values # Shape is (T*N)

    # CORRECTED: Reshape mX from (N, T, K) to match the (T*N, K) order.
    # We transpose the first two axes (N, T) -> (T, N) before reshaping.
    mX_stacked = mX.transpose(1, 0, 2).reshape(T * N, K)
    
    mX_with_const = sm.add_constant(mX_stacked)
    try:
        bp_test = sms.het_breuschpagan(residuals_stacked, mX_with_const)
        labels = ['Lagrange Multiplier Statistic', 'LM Test P-Value',
                  'F-Statistic', 'F-Test P-Value']
        
        print("Null Hypothesis (H0): Homoskedasticity (error variance is constant).")
        print(f"  LM Statistic: {bp_test[0]:.4f}")
        print(f"  LM Test P-Value: {bp_test[1]:.4f}")
        print(f"  F-Statistic: {bp_test[2]:.4f}")
        print(f"  F-Test P-Value: {bp_test[3]:.4f}")

        if bp_test[1] < 0.05 or bp_test[3] < 0.05:
            print("\nResult: Reject H0. Evidence of significant heteroskedasticity found.")
        else:
            print("\nResult: Fail to reject H0. No evidence of significant heteroskedasticity.")
    except Exception as e:
        print(f"Error running Breusch-Pagan test: {e}")
    
    print("-" * 55 + "\n")
    
    # --- VISUAL DIAGNOSTICS ---
    eps_sq = residuals.pow(2)
    
    # 1. Across-County Variance
    print("--- 1. Analyzing Across-County Variance (Unit Heteroskedasticity) ---")
    sigma_i2 = eps_sq.mean(axis=0)
    
    plt.figure(figsize=(8, 4))
    plt.hist(sigma_i2.dropna(), bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel("Estimated σ̂_i² (Within-County Residual Variance)")
    plt.ylabel("Number of Counties")
    plt.title("Distribution of County-Level Residual Variances")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    print("Summary of county-level residual variance (σ̂_i²):")
    print(sigma_i2.describe().to_string())
    print("-" * 55 + "\n")

    # 2. Over-Time Variance
    print("--- 2. Analyzing Over-Time Variance (Time Heteroskedasticity) ---")
    omega_t2 = eps_sq.mean(axis=1)
    
    plt.figure(figsize=(8, 4))
    plt.plot(omega_t2.index, omega_t2.values, marker='o', linestyle='-', markersize=4)
    plt.xlabel("Year")
    plt.ylabel("Estimated ω̂_t² (Cross-Sectional Residual Variance)")
    plt.title("Average Residual Variance Over Time")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    top_years = omega_t2.sort_values(ascending=False).head(5)
    print("Top 5 years with largest residual variance (ω̂_t²):")
    print(top_years.to_string())
    print("-" * 55 + "\n")
    
    # 3. Residual-by-Fitted-Value Plot
    print("--- 3. Analyzing Residual vs. Fitted Values ---")
    stack_kwargs = {'future_stack': True} if pd.__version__ >= '2.0' else {}
    
    resid_long = residuals.stack(**stack_kwargs).reset_index()
    resid_long.columns = ["time", "county", "residual"]
    fitted_long = fitted.stack(**stack_kwargs).reset_index()
    fitted_long.columns = ["time", "county", "fitted"]
    
    df_long = pd.merge(resid_long, fitted_long, on=["time", "county"])
    df_long["abs_resid"] = df_long["residual"].abs()
    
    plt.figure(figsize=(7, 6))
    hb = plt.hexbin(df_long["fitted"], df_long["abs_resid"], gridsize=50, cmap='viridis', mincnt=1)
    cb = plt.colorbar(hb)
    cb.set_label('Number of Observations')
    plt.xlabel("Fitted Value (ŷ_it)")
    plt.ylabel("Absolute Residual (|ε̂_it|)")
    plt.title("Residual vs. Fitted Value Plot")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    print("Generated residual-by-fitted-value plot to check for mean-dependent variance.")
    print("A 'fan' or 'funnel' shape suggests heteroskedasticity.")

if __name__ == '__main__':
    run_heteroskedasticity_diagnostics()