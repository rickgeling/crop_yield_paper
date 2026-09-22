import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.dates as mdates # If you want similar date formatting
from matplotlib.ticker import MultipleLocator

def calculate_time_varying_r_squared(
    mY_actual,    # Shape (T, N), actual (imputed) dependent variable
    mY_fitted,    # Shape (T, N), fitted values from the model
    years_axis,   # Shape (T,), for the x-axis of the plot (actual years)
    output_path,  # Path to save CSV and plot for this run
    model_run_label # String like "model_1_GS_state_17_run1" for filenames/titles
    ):
    """
    Calculates and plots time-varying R-squared (R_t^2).
    R_t^2 = 1 - (SSR_t / TSS_t) where SSR_t and TSS_t are calculated cross-sectionally at each time t.
    """
    print(f"Calculating Time-Varying R-squared for: {model_run_label}...")
    T_obs, N_units = mY_actual.shape
    r_squared_over_time = np.zeros(T_obs)

    # Decide on slicing. Original paper used [1:]. We'll use full range here.
    # If you want to match the paper's [1:] exactly, adjust loops and array indexing.
    # For example, loop t from 1 to T_obs-1 and store in r_squared_over_time[1:].
    
    for t in range(T_obs): # Loop over all time points
        residuals_t_sq = (mY_actual[t, :] - mY_fitted[t, :])**2
        ssr_t = np.sum(residuals_t_sq)
        
        mean_y_t = np.mean(mY_actual[t, :])
        total_variation_t_sq = (mY_actual[t, :] - mean_y_t)**2
        tss_t = np.sum(total_variation_t_sq)
        
        if tss_t == 0: 
            r_squared_over_time[t] = 1.0 if ssr_t == 0 else 0.0 # Or np.nan if preferred
        else:
            r_squared_over_time[t] = 1 - (ssr_t / tss_t)

    # Save R_t^2 to CSV
    # Assuming 'years_axis' corresponds to the time points 't'
    r_squared_t_df = pd.DataFrame({'year': years_axis, f'R_squared_t_{model_run_label}': r_squared_over_time})
    
    # Ensure "csv_data" subdirectory exists (or save directly to output_path)
    csv_dir = os.path.join(output_path, "csv_data_R2")
    os.makedirs(csv_dir, exist_ok=True)
    csv_filename = os.path.join(csv_dir, f"r_squared_time_varying_{model_run_label}.csv")
    try:
        r_squared_t_df.to_csv(csv_filename, index=False)
        print(f"Saved time-varying R-squared to: {csv_filename}")
    except Exception as e:
        print(f"Error saving time-varying R-squared CSV: {e}")

    # Plot R_t^2
    plt.style.use('seaborn-v0_8-whitegrid') # Or your preferred style
    fig, ax = plt.subplots(figsize=(12, 6)) # Use ax for date formatting
    ax.plot(years_axis, r_squared_over_time, marker='.', linestyle='-', label=f'$R_t^2$ for {model_run_label}')
    ax.set_title(f'Time-Varying R-squared ($R_t^2$)', fontsize="x-large")
    #ax.set_xlabel('Year')
    ax.set_ylabel('$R_t^2$', fontsize="x-large", labelpad=15)
    plt.tick_params(axis='both', labelsize=15)
    ax.grid(True, linestyle=':')
    #ax.legend()
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(10))
    #plt.xticks(rotation=45)
    
    # Optional: Date formatting similar to housing paper (if years_axis are actual years)
    # ax.xaxis.set_major_locator(mdates.YearLocator(5)) # Every 5 years
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    # fig.autofmt_xdate() 
    
    plt.tight_layout()
    
    # Ensure "figures" subdirectory exists
    figures_dir = os.path.join(output_path, "figures_R2")
    os.makedirs(figures_dir, exist_ok=True)
    plot_filename = os.path.join(figures_dir, f"r_squared_time_varying_{model_run_label}.eps")
    try:
        plt.savefig(plot_filename, format='eps')
        print(f"Saved time-varying R-squared plot to: {plot_filename}")
    except Exception as e:
        print(f"Error saving time-varying R-squared plot: {e}")
    plt.show()
    
    return r_squared_over_time



def calculate_pooled_r_squared(
    mY_actual,    # Shape (T, N)
    mY_fitted     # Shape (T, N)
    ):
    """
    Calculates the overall pooled R-squared.
    """
    print("Calculating Pooled R-squared...")
    residuals_sq = (mY_actual - mY_fitted)**2
    ssr_pooled = np.sum(residuals_sq)
    
    grand_mean_y = np.mean(mY_actual)
    total_variation_sq = (mY_actual - grand_mean_y)**2
    tss_pooled = np.sum(total_variation_sq)
    
    if tss_pooled == 0:
        r_squared_pooled = 1.0 if ssr_pooled == 0 else 0.0 # Or np.nan
    else:
        r_squared_pooled = 1 - (ssr_pooled / tss_pooled)
    
    print(f"Pooled R-squared: {r_squared_pooled:.4f}")
    return r_squared_pooled



def calculate_per_region_r_squared(
    mY_actual,    # Shape (T, N)
    mY_fitted,    # Shape (T, N)
    regions,      # List of N region names
    output_path,
    model_run_label,
    plot_results=True # New argument to control plotting
    ):
    """
    Calculates R-squared for each region/unit over time (R_i^2).
    R_i^2 = 1 - (SSR_i / TSS_i) where SSR_i and TSS_i are calculated over time for each region i.
    Saves the R_i^2 values to a CSV and optionally plots them.
    """
    print(f"Calculating Per-Region R-squared for: {model_run_label}...")
    T_obs, N_units = mY_actual.shape
    r_squared_per_region = np.zeros(N_units)

    # As before, decide if you need to omit the first time point like the housing paper (e.g., mY_actual[1:, i])
    # For this example, we'll use the full time series for each region.
    for i in range(N_units): # Loop over regions
        residuals_i_sq = (mY_actual[:, i] - mY_fitted[:, i])**2
        ssr_i = np.sum(residuals_i_sq)
        
        mean_y_i = np.mean(mY_actual[:, i])
        total_variation_i_sq = (mY_actual[:, i] - mean_y_i)**2
        tss_i = np.sum(total_variation_i_sq)
        
        if tss_i == 0:
            r_squared_per_region[i] = 1.0 if ssr_i == 0 else 0.0 
        else:
            r_squared_per_region[i] = 1 - (ssr_i / tss_i)

    r_squared_i_df = pd.DataFrame({
        'region_name': regions, # Using actual region names
        f'R_squared_i': r_squared_per_region
    })
    # Sort by R-squared value for better visualization in tables or some plot types
    r_squared_i_df = r_squared_i_df.sort_values(by=f'R_squared_i', ascending=True) 
    
    csv_dir = os.path.join(output_path, "csv_data_R2")
    os.makedirs(csv_dir, exist_ok=True)
    csv_filename = os.path.join(csv_dir, f"r_squared_per_region_{model_run_label}.csv")
    try:
        r_squared_i_df.to_csv(csv_filename, index=False)
        print(f"Saved per-region R-squared to: {csv_filename}")
    except Exception as e:
        print(f"Error saving per-region R-squared CSV: {e}")

    if plot_results:
        print(f"Plotting Per-Region R-squared for: {model_run_label}...")
        # For plotting, if N_units is very large, a line plot against index might be better.
        # If N_units is moderate, a bar chart can be effective.
        # Let's do a line plot against a simple index for now, similar to housing paper.
        
        # Use sorted values for a potentially more informative line plot
        plot_idx = np.arange(1, N_units + 1)
        sorted_r_squared_values = r_squared_i_df[f'R_squared_i'].values # Already sorted

        # sns.set_theme() # Optional: if we want seaborn styling
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        
        plt.plot(plot_idx, sorted_r_squared_values, marker='o', linestyle='-', markersize=4,
                 label=f'$R_i^2$ for {model_run_label}')
        
        plt.ylabel(r'$R_i^2$ Value (Sorted)', fontsize="x-large", labelpad=15)
        plt.xlabel('Counties (Sorted by $R_i^2$)', fontsize="x-large", labelpad=15)
        plt.title(f'$R^2$ over Regions (Sorted)', fontsize="x-large")
        plt.grid(True, linestyle=':')
        plt.tick_params(axis='both', labelsize=15)
        #plt.legend(fontsize='medium', loc='best')
        plt.tight_layout()
        
        figures_dir = os.path.join(output_path, "figures_R2")
        os.makedirs(figures_dir, exist_ok=True)
        plot_filename = os.path.join(figures_dir, f"r_squared_per_region_{model_run_label}.eps")
        try:
            plt.savefig(plot_filename, format='eps')
            print(f"Saved per-region R-squared plot to: {plot_filename}")
        except Exception as e:
            print(f"Error saving per-region R-squared plot: {e}")
        plt.show()
    
    return r_squared_i_df





# --- MAIN:
def run_all_r_squared_analyses(
    mY_actual, mY_fitted, years_axis, regions, 
    output_path, model_run_label
    ):
    print(f"\n--- Starting All R-squared Analyses for: {model_run_label} ---")
    
    r_t_sq_series = calculate_time_varying_r_squared(mY_actual, mY_fitted, years_axis, output_path, model_run_label)
    pooled_r_sq_value = calculate_pooled_r_squared(mY_actual, mY_fitted)
    r_i_sq_dataframe = calculate_per_region_r_squared(mY_actual, mY_fitted, regions, output_path, model_run_label, \
        plot_results=True)
    
    print(f"--- Completed All R-squared Analyses for: {model_run_label} ---")
    return {
        "time_varying_r2_series": r_t_sq_series, 
        "pooled_r2_value": pooled_r_sq_value, 
        "per_region_r2_df": r_i_sq_dataframe
    }
















##################_-----------------------------------------


# In methods/lldve_evaluation.py

import numpy as np
import pandas as pd # Though not strictly needed if mY and mY_fitted are arrays
import matplotlib.pyplot as plt
import seaborn as sns # Used by the original R2() function's plotting
import matplotlib.dates as mdates # Used for date formatting in plots
import os

def replicate_housing_paper_R2_logic(
    mY_actual,          # Shape (T, N), your actual (imputed) log yields
    mY_fitted,          # Shape (T, N), fitted values from YOUR LLDVE model run
    years_axis,         # Shape (T,), actual years for plotting (replaces 'idx' from Dates.csv)
    output_path,        # Base path for saving outputs for this run
    model_run_label     # String like "model_1_GS_state_17_run1" for filenames/titles
    ):
    """
    Replicates the R2 calculations from the housing paper's R2() function,
    applying its specific logic (including slicing and averaging in intermediate steps)
    to the provided mY_actual and mY_fitted data from a single model run.

    Saves plots and prints pooled R-squared.
    """
    print(f"--- Replicating Housing Paper R2 Logic for: {model_run_label} ---")

    # Ensure "figures_R2_replicated" subdirectory exists
    figures_dir = os.path.join(output_path, "figures_R2_replicated")
    os.makedirs(figures_dir, exist_ok=True)

    # Get dimensions from your data
    T_obs, N_units = mY_actual.shape

    # --- Time-Varying R-squared (R_t^2) ---
    # Mimicking the variable names and logic from the housing paper's R2() for one model
    vRSS_model = np.zeros(T_obs)  # Residual Sum of Squares (scaled) for your model at each time t
    vSST_time = np.zeros(T_obs)   # Total Sum of Squares (scaled) at each time t
    
    for t in range(T_obs):
        # SSR for your model at time t (sum across regions), scaled by N_units
        vRSS_model[t] = np.sum((mY_fitted[t, :] - mY_actual[t, :])**2) / N_units
        # TSS at time t (sum across regions, variation around cross-sectional mean at time t), scaled by N_units
        vSST_time[t] = np.sum((mY_actual[t, :] - np.mean(mY_actual[t, :]))**2) / N_units

    # Replicating the intermediate plot from the original R2() if desired
    # This plotted mean(mY_hat_base), mean(mY_hat_full), mean(mY)[1:]
    # For a single model, we can plot mean(mY_fitted) and mean(mY_actual)
    plt.figure(figsize=(8,5)) # Smaller figure for this intermediate plot
    plt.plot(years_axis, np.mean(mY_fitted, axis=1), label=r'$\overline{\hat{y}}_t$ (Fitted)', linestyle='--')
    # Applying [1:] slicing if mimicking the original plot's Y actual exactly, otherwise use full years_axis
    if T_obs > 1 : # Ensure there's something to slice
        plt.plot(years_axis[1:], np.mean(mY_actual, axis=1)[1:], label=r'$\overline{y}_t$ (Actual, from t=1)', linestyle='-')
    else:
        plt.plot(years_axis, np.mean(mY_actual, axis=1), label=r'$\overline{y}_t$ (Actual)', linestyle='-')
    plt.title(f'Mean Actual vs. Mean Fitted Y over Time\n{model_run_label}')
    plt.xlabel('Year')
    plt.ylabel('Mean Log Yield')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Calculate R^2 for each time t, applying the [1:] slicing as in the original paper's code
    if T_obs > 1: # Ensure slicing is possible
        vR2_time_model = 1 - vRSS_model[1:] / vSST_time[1:]
        plot_years_axis = years_axis[1:]
    else: # Handle case with only one time point (R2 will be single value, or handle error)
        vR2_time_model = np.array([1 - vRSS_model[0] / vSST_time[0]]) if vSST_time[0] != 0 else np.array([0.0])
        plot_years_axis = years_axis

    sns.set_theme() # As in original R2()
    plt.figure(figsize=(10,6)) # Standardized figure size
    ax = plt.gca()
    plt.plot(plot_years_axis, vR2_time_model, c='b', label=f'{model_run_label}')
    plt.ylabel(r'$R^2$')
    plt.title(r'$R^2$ over Time (Replicating Housing Paper Logic)')
    
    # Date formatting (assuming years_axis contains numerical years)
    # For numerical years, mdates might not be ideal unless converted to datetime objects.
    # Simple year display is often fine for annual data.
    # If you want exact date formatting, ensure years_axis is array of datetime objects.
    # Example: ax.xaxis.set_major_locator(mdates.YearLocator(5)) # Every 5 years
    #          ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    #          plt.gcf().autofmt_xdate() 

    plt.xlabel('Year') # Simplified x-label
    plt.legend(fontsize='medium', loc='best')
    plt.xticks(fontsize=13, rotation=0) # Adjusted
    plt.yticks(fontsize=13) # Adjusted
    plt.grid(True, linestyle=':') # Added grid for readability
    plt.tight_layout()
    
    plot_filename_time_rep = os.path.join(figures_dir, f"replicated_r2_time_{model_run_label}.eps")
    try:
        plt.savefig(plot_filename_time_rep, format='eps')
        print(f"Saved replicated time-varying R2 plot to: {plot_filename_time_rep}")
    except Exception as e:
        print(f"Error saving replicated time-varying R2 plot: {e}")
    plt.show()

    # --- R-squared over regions ($R_i^2$) ---
    # Replicating the variable reuse style for RSS and SST for regions
    vRSS_model_region = np.zeros(N_units)
    vSST_region = np.zeros(N_units)  
    
    for i in range(N_units): # Loop over regions
        # Applying [1:] slicing for time as in the original R2() function
        if T_obs > 1:
            vRSS_model_region[i] = np.sum((mY_fitted[1:, i] - mY_actual[1:, i])**2) / (T_obs -1 if T_obs > 1 else 1)
            vSST_region[i] = np.sum((mY_actual[1:, i] - np.mean(mY_actual[1:, i]))**2) / (T_obs -1 if T_obs > 1 else 1)
        else: # Handle case with only one time point
            vRSS_model_region[i] = np.sum((mY_fitted[:, i] - mY_actual[:, i])**2) 
            vSST_region[i] = 0 # TSS is 0 if only one data point per region for variance calc
                               # R2 would be undefined or 0. Original paper had T=180.

    # Avoid division by zero for regions where TSS_region might be zero
    vR2_per_region_model = np.zeros_like(vSST_region)
    for i in range(N_units):
        if vSST_region[i] != 0:
            vR2_per_region_model[i] = 1 - vRSS_model_region[i] / vSST_region[i]
        elif vRSS_model_region[i] == 0: # If TSS is 0 and SSR is 0, R2 is 1
            vR2_per_region_model[i] = 1.0
        else: # If TSS is 0 and SSR is not 0, R2 is undefined or -infinity, set to 0 or NaN
            vR2_per_region_model[i] = 0.0 # Or np.nan

    idx_region_plot = np.arange(1, N_units + 1, 1) # x-axis for region plot
    sns.set_theme() # As in original R2()
    plt.figure(figsize=(10,6)) # Standardized figure size
    ax = plt.gca()
    plt.plot(idx_region_plot, vR2_per_region_model, c='b', label=f'{model_run_label}', marker='.', linestyle='-')
    plt.ylabel(r'$R^2$')
    plt.xlabel('Region Index') # Original used no label, used index
    plt.title(r'$R^2$ over Regions (Replicating Housing Paper Logic)')
    plt.legend(fontsize='medium', loc='best')
    plt.xticks(fontsize=13, rotation=0) # Adjusted
    plt.yticks(fontsize=13) # Adjusted
    plt.grid(True, linestyle=':') # Added grid for readability
    plt.tight_layout()
    
    plot_filename_region_rep = os.path.join(figures_dir, f"replicated_r2_region_{model_run_label}.eps")
    try:
        plt.savefig(plot_filename_region_rep, format='eps')
        print(f"Saved replicated per-region R2 plot to: {plot_filename_region_rep}")
    except Exception as e:
        print(f"Error saving replicated per-region R2 plot: {e}")
    plt.show()

    # --- Pooled Panel R-squared ---
    dPanelR2_model = 1 - np.sum((mY_actual - mY_fitted)**2) / np.sum((mY_actual - np.mean(mY_actual))**2)
    print(f"Replicated Pooled R-squared ({model_run_label}): {dPanelR2_model:.4f}")
    
    print(f"--- Completed Replication of Housing Paper R2 Logic for: {model_run_label} ---")
    
    return {
        "time_varying_r2": vR2_time_model if T_obs > 1 else vR2_time_model[0],
        "per_region_r2": vR2_per_region_model,
        "pooled_r2": dPanelR2_model,
        "time_axis_for_time_r2": plot_years_axis if T_obs > 1 else years_axis
    }




################# ------------------------------------------------------------------


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.dates as mdates # If using date formatting
import matplotlib.transforms as transforms # For text on plot as in original

def calculate_explained_variation_decomposition(
    mTheta_hat,         # Shape (T, num_total_params_incl_global_trend)
    mX_regressors,      # Shape (N, T, num_weather_vars), from model_specification
    vAlpha_hat,         # Shape (N,), estimated fixed effects
    years_axis,         # Shape (T,), for the x-axis of the plot (original time points)
    output_path,        # Path to save CSV and plot for this run
    model_run_label,    # String for unique filenames/titles
    block_average_periods=[3, 6] # e.g., [3, 6] for 3Y and 6Y averages
    ):
    """
    Calculates and plots the explained variation decomposition, showing contributions
    of global trend vs. weather components, with different averaging periods.
    """
    print(f"Calculating Explained Variation Decomposition for: {model_run_label}...")

    T_obs = mTheta_hat.shape[0]
    N_units = vAlpha_hat.shape[0]

    if T_obs <= 1: # Need at least 2 time periods to calculate differences
        print("Cannot calculate variation decomposition with T_obs <= 1. At least 2 time periods are required.")
        return None

    # 1. Global Trend Component (g(t))
    vGlobal_trend_t = mTheta_hat[:, 0]

    # 2. Average Local Component (L_bar_t = average of [alpha_i + weather_effects_it])
    weather_effects_Nt = np.zeros((N_units, T_obs))
    for i in range(N_units):
        for t in range(T_obs):
            # mX_regressors[i, t, :] is (d_weather_vars,)
            # mTheta_hat[t, 1:] is (d_weather_vars,)
            weather_effects_Nt[i, t] = mX_regressors[i, t, :] @ mTheta_hat[t, 1:]
            
    local_component_Nt = weather_effects_Nt + vAlpha_hat[:, np.newaxis] # Broadcast vAlpha_hat
    vAverage_local_component_t = np.mean(local_component_Nt, axis=0) # Shape (T_obs,)

    # 3. Absolute Differences (Deltas) - these will have length T_obs-1
    # The x-axis for these differences/ratios will be years_axis[1:]
    years_for_ratios = years_axis[1:] 
    
    vGlobal_trend_diff = np.abs(vGlobal_trend_t[1:] - vGlobal_trend_t[:-1])
    vAverage_local_component_diff = np.abs(vAverage_local_component_t[1:] - vAverage_local_component_t[:-1])

    # 4. Explained Variation Ratios (length T_obs-1)
    denominator = vGlobal_trend_diff + vAverage_local_component_diff
    
    vExplRatio_Global = np.zeros_like(denominator)
    vExplRatio_Weather = np.zeros_like(denominator)
    
    valid_indices = denominator != 0
    both_zero_diff_indices = (vGlobal_trend_diff == 0) & (vAverage_local_component_diff == 0)

    vExplRatio_Global[valid_indices] = (vGlobal_trend_diff[valid_indices] / denominator[valid_indices]) * 100
    vExplRatio_Weather[valid_indices] = (vAverage_local_component_diff[valid_indices] / denominator[valid_indices]) * 100
    
    # Handle 0/0 case: if both diffs are 0, assign 50/50 or NaN. Let's use 50 for now.
    # (This also covers where denominator is 0 because both diffs are 0)
    vExplRatio_Global[both_zero_diff_indices] = 50 
    vExplRatio_Weather[both_zero_diff_indices] = 50


    # --- Save 1Y (raw annual) Ratios to CSV ---
    csv_dir = os.path.join(output_path, "csv_data_ExplVar")
    os.makedirs(csv_dir, exist_ok=True)
    
    explained_var_1Y_df = pd.DataFrame({
        'year': years_for_ratios, # Correct x-axis for these T-1 ratios
        f'ExplVar_GlobalTrend_1Y': vExplRatio_Global, # Simplified column name
        f'ExplVar_Weather_1Y': vExplRatio_Weather
    })
    csv_filename_1Y = os.path.join(csv_dir, f"explained_variation_1Y_ratios_{model_run_label}.csv")
    try:
        explained_var_1Y_df.to_csv(csv_filename_1Y, index=False)
        print(f"Saved 1Y explained variation ratios to: {csv_filename_1Y}")
    except Exception as e:
        print(f"Error saving 1Y explained variation CSV: {e}")

    # --- Calculate and Save Block Averages ---
    dict_for_final_df_return = {
        "years_1Y_ratio": years_for_ratios,
        "global_trend_1Y": vExplRatio_Global,
        "weather_1Y": vExplRatio_Weather,
    }
    
    # Store calculated averages for plotting
    plot_avg_ratios_global = {}
    plot_avg_ratios_weather = {}
    plot_avg_years = {}

    for period_Y in block_average_periods: # e.g., 3 for 3Y, 6 for 6Y
        if len(vExplRatio_Global) >= period_Y: # Check if series is long enough for this block
            num_blocks = len(vExplRatio_Global) // period_Y
            if num_blocks == 0: continue # Not enough data for even one full block

            avg_g = np.zeros(num_blocks)
            avg_w = np.zeros(num_blocks)
            avg_y_axis = np.zeros(num_blocks, dtype=years_for_ratios.dtype) # Match type of years

            for i_block in range(num_blocks):
                start_idx = i_block * period_Y
                end_idx = start_idx + period_Y
                avg_g[i_block] = np.mean(vExplRatio_Global[start_idx:end_idx])
                avg_w[i_block] = np.mean(vExplRatio_Weather[start_idx:end_idx])
                avg_y_axis[i_block] = years_for_ratios[start_idx] # Year at start of block
            
            period_Y_str = f'{period_Y}Y'
            plot_avg_ratios_global[period_Y_str] = avg_g
            plot_avg_ratios_weather[period_Y_str] = avg_w
            plot_avg_years[period_Y_str] = avg_y_axis
            
            # Add to dictionary for final DataFrame output
            dict_for_final_df_return[f"years_{period_Y_str}_avg"] = avg_y_axis
            dict_for_final_df_return[f"global_trend_{period_Y_str}"] = avg_g
            dict_for_final_df_return[f"weather_{period_Y_str}"] = avg_w

            # Save these block averages to separate CSVs
            temp_df_avg = pd.DataFrame({
                f'year_start_{period_Y_str}': avg_y_axis, 
                f'ExplVar_GlobalTrend_{period_Y_str}': avg_g,
                f'ExplVar_Weather_{period_Y_str}': avg_w
            })
            avg_csv_filename = os.path.join(csv_dir, f"explained_variation_{period_Y_str}_avg_{model_run_label}.csv")
            try:
                temp_df_avg.to_csv(avg_csv_filename, index=False)
                print(f"Saved {period_Y_str} explained variation ratios to: {avg_csv_filename}")
            except Exception as e:
                print(f"Error saving {period_Y_str} explained variation CSV: {e}")
        else:
            print(f"Time series too short for {period_Y}Y averaging (length: {len(vExplRatio_Global)}, needed: {period_Y}).")


    # --- Plotting ---
    figures_dir = os.path.join(output_path, "figures_ExplVar")
    os.makedirs(figures_dir, exist_ok=True)
    
    plot_configurations = [
        (vExplRatio_Global, plot_avg_ratios_global, "Global Trend", "trend"),
        (vExplRatio_Weather, plot_avg_ratios_weather, "Weather Components", "weather")
    ]

    for vExplRatio_1Y, dict_avg_ratios_for_plot, title_component, file_suffix_comp in plot_configurations:
        dAverage_explained_1Y = np.mean(vExplRatio_1Y)
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(12, 7))
        ax = plt.gca()
        
        plt.plot(years_for_ratios, vExplRatio_1Y, linewidth=1.5, alpha=0.7, label='1Y Ratio (Annual)')
        for period_label, avg_data in dict_avg_ratios_for_plot.items(): # period_label is '3Y', '6Y'
            plt.plot(plot_avg_years[period_label], avg_data, linewidth=1.5, linestyle='-', marker='o', markersize=4, label=f'{period_label} Avg.')
            
        plt.axhline(y=dAverage_explained_1Y, color='r', linestyle=':', linewidth=1.5, label=f'Mean 1Y Ratio ({dAverage_explained_1Y:.0f}%)')
        # For text inside plot, adjust position if needed
        # trans = transforms.blended_transform_factory(ax.get_yticklabels()[0].get_transform(), ax.transData)
        # ax.text(0.01, dAverage_explained_1Y + 2, f"Avg: {dAverage_explained_1Y:.0f}%", 
        #         color="red", transform=trans, ha="left", va="bottom", fontsize=10)

        plt.title(f'Explained Variation (in %) by {title_component}', fontsize="x-large")
        plt.ylabel(f'% Explained by {title_component}', fontsize="x-large", labelpad=15)
        plt.ylim(-5, 105) 
        plt.legend(fontsize='medium', loc='best')
        plt.tick_params(axis='both', labelsize=15)
        #plt.xticks(fontsize=13)
        #plt.yticks(fontsize=13)
        plt.grid(True, linestyle=':')
        ax = plt.gca()
        ax.xaxis.set_major_locator(MultipleLocator(10))
        plt.tight_layout()
        
        plot_filename = os.path.join(figures_dir, f"explained_variation_{file_suffix_comp}_{model_run_label}.eps")
        try:
            plt.savefig(plot_filename, format='eps')
            print(f"Saved explained variation plot ({title_component}) to: {plot_filename}")
        except Exception as e:
            print(f"Error saving explained variation plot ({title_component}): {e}")
        plt.show()
        
    # Create a single DataFrame from the dictionary of all series (handles different lengths with NaNs)
    final_df_to_return = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in dict_for_final_df_return.items()]))
    
    return final_df_to_return