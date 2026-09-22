
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def calculate_pooled_r_squared(mY_actual, mY_fitted_np):
    """
    Calculates pooled R-squared using the exact reference methodology.
    """
    y_actual_flat = mY_actual.flatten()
    y_fitted_flat = mY_fitted_np.flatten()
    
    # Using np.sum as requested
    ssr = np.sum((y_actual_flat - y_fitted_flat)**2)
    tss = np.sum((y_actual_flat - np.mean(y_actual_flat))**2)
    
    # R-squared calculated directly, without if/else check
    return 1 - (ssr / tss)

def calculate_per_region_r_squared(mY_actual, mY_fitted_np, regions, output_path, model_run_label):
    """
    Calculates R-squared for each region using the user's specific formula.
    """
    print(f"Calculating Per-Region R-squared for: {model_run_label}...")
    T_obs, N_units = mY_actual.shape
    
    # Using variable names and logic from housing reference code
    vRSS = np.zeros(N_units)
    vSST = np.zeros(N_units)

    for i in range(N_units):
        # Slicing with [1:] and dividing by (T-1) to match the reference
        y_actual_sliced = mY_actual[1:, i]
        y_fitted_sliced = mY_fitted_np[1:, i]
        
        # Using np.sum and dividing by the number of observation
        vRSS[i] = np.sum((y_fitted_sliced - y_actual_sliced)**2) / (T_obs - 1)
        vSST[i] = np.sum((y_actual_sliced - np.mean(y_actual_sliced))**2) / (T_obs - 1)

    # Final R-squared calculation
    vR2 = 1 - (vRSS / vSST)
    
    r2_df = pd.DataFrame({'region_name': regions, 'R_Squared': vR2}).sort_values(by='R_Squared', ascending=False)
    
    # Plotting
    plt.figure(figsize=(12, 7)); plt.plot(np.arange(N_units), r2_df['R_Squared'], 'o-'); plt.title('Per-Region R-Squared'); plt.ylabel('R-Squared'); plt.grid(True, linestyle=':');
    plt.savefig(os.path.join(output_path, f"per_region_r2_{model_run_label}.png"), dpi=300); plt.show(); plt.close()

def calculate_time_varying_r_squared(mY_actual, mY_fitted_np, years, output_path, model_run_label):
    """
    Calculates R-squared for each time period using the user's specific formula.
    """
    print(f"Calculating Time-Varying R-squared for: {model_run_label}...")
    T_obs, N_units = mY_actual.shape
    
    vRSS = np.zeros(T_obs)
    vSST = np.zeros(T_obs)

    for t in range(T_obs):
        # Using np.sum and dividing by the number of observation
        vRSS[t] = np.sum((mY_fitted_np[t, :] - mY_actual[t, :])**2) / N_units
        vSST[t] = np.sum((mY_actual[t, :] - np.mean(mY_actual[t, :]))**2) / N_units
        
    # Final R-squared calculation, slicing with [1:] as per the housing reference
    vR2 = 1 - (vRSS[1:] / vSST[1:])

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6)); ax.plot(years[1:], vR2, '.-'); ax.set_title('Time-Varying R-Squared');
    ax.set_ylabel('R-Squared'); ax.xaxis.set_major_locator(MultipleLocator(10)); ax.grid(True, linestyle=':');
    plt.savefig(os.path.join(output_path, f"time_varying_r2_{model_run_label}.png"), dpi=300); plt.show(); plt.close()

def run_all_spline_r_squared_analyses(
    mY_actual, spline_results, spline_df_to_run,
    years, regions, output_path, model_run_label
    ):
    """
    """
    print("\n" + "="*70)
    print(f"Running R-Squared Evaluation Suite for Spline (DF={spline_df_to_run})")
    print("="*70)

    mY_fitted_df = spline_results[spline_df_to_run].get('fitted_values_panel')
    if mY_fitted_df is None:
        print(f"Error: 'fitted_values_panel' not found. Skipping R2 analysis.")
        return

    mY_fitted_np = mY_fitted_df.values
    if mY_actual.shape != mY_fitted_np.shape:
        print("Error: Shape mismatch. Skipping R2 analysis.")
        return

    r2_output_path = os.path.join(output_path, f"R_Squared_Analysis_DF{spline_df_to_run}")
    os.makedirs(r2_output_path, exist_ok=True)
    
    pooled_r2 = calculate_pooled_r_squared(mY_actual, mY_fitted_np)
    print(f"  Pooled R-squared: {pooled_r2:.4f}")
    with open(os.path.join(r2_output_path, f"pooled_r2_{model_run_label}.txt"), "w") as f:
        f.write(f"Pooled R-squared: {pooled_r2}\n")

    calculate_per_region_r_squared(mY_actual, mY_fitted_np, regions, r2_output_path, model_run_label)
    calculate_time_varying_r_squared(mY_actual, mY_fitted_np, years, r2_output_path, model_run_label)

    print("\n--- R-Squared Evaluation Complete ---")