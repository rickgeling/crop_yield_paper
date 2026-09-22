


# # FIX DIRECTORY (we are in parent/post_diagnostics, we need to get LLDVE_new from parent/methods)
# from methods import LLDVE_new

# mY = 
# mX = 
# # GET mTHETA stuff for H_opt
# mTheta_hat_opt = load...

# T, N   = mY.shape
# t_i    = np.arange(1, T+1) / T
# h_opt = np.load(h_opt)
# h_low = h_opt*0.9 #(- 10%)
# h_high = h_opt*1.1 #(+ 10%)

# # LOW:
# mK_opt        = np.asarray([estK(t_i - i/T, h_low) for i in range(1, T+1)])
# mTheta_hat, vAlpha_hat = est_mdl(mY, mX, t_i, T, N, mK_opt)


# # HIGH
# mK_opt_low        = np.asarray([estK(t_i - i/T, h_low) for i in range(1, T+1)])
# mTheta_hat_low, vAlpha_hat_low = est_mdl(mY, mX, t_i, T, N, mK_opt)


# #PLOT ALL 3 TOGETHER:

# bandwidth_sensitivity.py

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# results_config.py lives in the repo root, two folders up from here
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import results_config as cfg

# --- estK and est_mdl come from methods/LLDVE_test.py (there is no LLDVE_new.py) ---
try:
    from methods.LLDVE_test import estK, est_mdl
except ImportError:
    print("Error: could not import estK / est_mdl from methods/LLDVE_test.py.")
    sys.exit(1)


def run_bandwidth_sensitivity_test():
    """
    Loads core data, re-estimates coefficients for three different bandwidths,
    calculates quantitative sensitivity metrics, and plots them for comparison.
    """
    # ── which run to diagnose ──────────────────────────────────────────────
    # The paper's main run: corn, the four rainfed states pooled. The dataset
    # and the folder layout come from results_config.py in the repo root.
    final_res_subfolder = "17_18_19_27_model_3_GS_logY_anomalyX"
    results_dir = cfg.run_dir(final_res_subfolder)
    
    print(f"Loading results from: {results_dir}\n")

    h_path = os.path.join(results_dir, "h_optimal.txt")
    y_path = os.path.join(results_dir, "input_mY.csv")
    x_path = os.path.join(results_dir, "input_mX.npy")
    regions_path = os.path.join(results_dir, "regions_list.csv")
    
    required_files = [h_path, y_path, x_path, regions_path]
    if not all(os.path.exists(p) for p in required_files):
        print("Error: One or more required input files not found.")
        sys.exit(1)

    h_opt = float(np.loadtxt(h_path))
    mY_full = pd.read_csv(y_path, index_col=0)
    mX = np.load(x_path)
    df_regions = pd.read_csv(regions_path)
    
    model_counties = df_regions['region_name'].tolist() if 'region_name' in df_regions.columns else df_regions.iloc[:, 0].tolist()
    mY_aligned_df = mY_full[model_counties]
    mY = mY_aligned_df.values
    years = mY_aligned_df.index.to_numpy()
    
    print("Data successfully loaded and aligned.")
    
    # --- 2. Define Bandwidths and Re-estimate ALL Paths ---
    N, T, K = mX.shape
    t_i = np.arange(1, T + 1) / T
    h_low = h_opt * 0.9
    h_high = h_opt * 1.1

    print(f"Optimal h = {h_opt:.3f}. Re-estimating for h_opt, h_low={h_low:.3f}, and h_high={h_high:.3f}")

    print("Estimating with optimal bandwidth...")
    mK_opt = np.asarray([estK(t_i - i/T, h_opt) for i in range(1, T+1)])
    mTheta_hat_opt, _ = est_mdl(mY, mX, t_i, T, N, mK_opt)

    print("Estimating with low bandwidth...")
    mK_low = np.asarray([estK(t_i - i/T, h_low) for i in range(1, T+1)])
    mTheta_hat_low, _ = est_mdl(mY, mX, t_i, T, N, mK_low)

    print("Estimating with high bandwidth...")
    mK_high = np.asarray([estK(t_i - i/T, h_high) for i in range(1, T+1)])
    mTheta_hat_high, _ = est_mdl(mY, mX, t_i, T, N, mK_high)

    num_coefficients = mTheta_hat_opt.shape[1]
    
    # --- 3. Quantitative Sensitivity Metrics ---
    print("\n" + "-"*60)
    print("Quantitative Bandwidth Sensitivity Analysis")
    print("-" * 60)

    for j in range(num_coefficients):
        beta_opt = mTheta_hat_opt[:, j]
        beta_low = mTheta_hat_low[:, j]
        beta_high = mTheta_hat_high[:, j]

        # --- Calculate Deviations ---

        # Maximum Absolute Deviation (MAD)
        mad_low = np.max(np.abs(beta_low - beta_opt))
        mad_high = np.max(np.abs(beta_high - beta_opt))

        # Root Mean Square Deviation (RMSD)
        rmsd_low = np.sqrt(np.mean((beta_low - beta_opt)**2))
        rmsd_high = np.sqrt(np.mean((beta_high - beta_opt)**2))
        
        # --- Calculate Percentage Deviation ---
        
        # Get the range of the optimal coefficient path to use as a baseline
        beta_range = np.max(beta_opt) - np.min(beta_opt)
        
        # Avoid division by zero if the coefficient path is flat
        if beta_range < 1e-9:
            mad_low_pct = np.nan # Or 0, depending on preference
            mad_high_pct = np.nan
        else:
            mad_low_pct = (mad_low / beta_range) * 100
            mad_high_pct = (mad_high / beta_range) * 100

        # --- Print Results ---
        
        print(f"\n--- Coefficient {j+1} ---")
        print(f"  Deviation from h-10% (h={h_low:.3f}):")
        print(f"    - Max Abs Deviation (in coeff units): {mad_low:.4f}")
        print(f"    - Max Abs Deviation (as % of range): {mad_low_pct:.2f}%")
        print(f"    - Root Mean Square Deviation:         {rmsd_low:.4f}")
        print(f"  Deviation from h+10% (h={h_high:.3f}):")
        print(f"    - Max Abs Deviation (in coeff units): {mad_high:.4f}")
        print(f"    - Max Abs Deviation (as % of range): {mad_high_pct:.2f}%")
        print(f"    - Root Mean Square Deviation:         {rmsd_high:.4f}")
    
    print("\n" + "-"*60 + "\n")


    # --- 4. Plot All Three Coefficient Paths Together ---
    print("Generating comparison plots...")
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    for j in range(num_coefficients):
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(years, mTheta_hat_opt[:, j], color='royalblue', linewidth=2.5, 
                label=f'$h_{{opt}} = {h_opt:.3f}$')
        ax.plot(years, mTheta_hat_low[:, j], color='darkgreen', linestyle='--', linewidth=1.5, 
                label=f'$h = {h_low:.3f}$ (-10%)')
        ax.plot(years, mTheta_hat_high[:, j], color='firebrick', linestyle=':', linewidth=1.5, 
                label=f'$h = {h_high:.3f}$ (+10%)')
        
        ax.set_title(f"Bandwidth Sensitivity for Coefficient {j+1}", fontsize=16)
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel(f"Coefficient Value", fontsize=12)
        ax.legend(loc='best', title="Bandwidth", fontsize=10)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.axhline(0, color='black', lw=0.75)
        
        plt.tight_layout()
        
        plot_filename = os.path.join(results_dir, f"bandwidth_sensitivity_coeff_{j+1}.png")
        try:
            plt.savefig(plot_filename, format='png', dpi=300)
            print(f"  - Plot for Coefficient {j+1} saved to: {os.path.basename(plot_filename)}")
        except Exception as e:
            print(f"\nError saving plot for Coefficient {j+1}: {e}")
            
        plt.show()


if __name__ == '__main__':
    run_bandwidth_sensitivity_test()