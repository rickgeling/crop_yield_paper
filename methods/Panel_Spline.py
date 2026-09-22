
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import patsy
import matplotlib.pyplot as plt
import re
from matplotlib.ticker import MultipleLocator
import warnings
from statsmodels.tools.sm_exceptions import ValueWarning
warnings.filterwarnings("ignore", category=ValueWarning, message=".*covariance of constraints.*")

def _sanitize_name_for_patsy(name):
    """Cleans a variable name to be patsy-compatible for column naming."""
    name = name.replace('²', '_sq')
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip('_')
    return name

def _prepare_long_format_df(mY, mZ, years, regions, var_names):
    """Converts panel arrays into a long-format DataFrame and sanitizes column names."""
    N_z, T_z, K_z = mZ.shape
    
    if mY.shape == (N_z, T_z):
        mY_corrected = mY
    elif mY.shape == (T_z, N_z):
        mY_corrected = mY.T
    else:
        raise ValueError(f"Fatal Shape Mismatch: mZ has shape (N, T) = ({N_z}, {T_z}), but mY has an incompatible shape {mY.shape}.")
        
    N, T = mY_corrected.shape
    assert len(regions) == N and len(years) == T and K_z == len(var_names)

    sanitized_names = [_sanitize_name_for_patsy(name) for name in var_names]
    name_map = dict(zip(var_names, sanitized_names))
    
    df_list = []
    for i in range(N):
        for t in range(T):
            row = {'yield': mY_corrected[i, t], 'year': years[t], 'region': regions[i]}
            for k, s_name in enumerate(sanitized_names):
                row[s_name] = mZ[i, t, k]
            df_list.append(row)
            
    df = pd.DataFrame(df_list)
    df = df.dropna(subset=['yield'])
    return df, name_map

def _plot_time_varying_coefficient_manual(results, spline_df, weather_s_name, year_range, output_path, title_name):
    """
    Manually calculates and plots a single time-varying weather coefficient.
    NOW RETURNS the calculated coefficients and CIs.
    """
    params = results.params
    cov_matrix = results.cov_params()
    param_names = params.index

    main_effect_coef = params.get(weather_s_name, 0)
    interaction_coef_names = [name for name in param_names if f"*{weather_s_name}" in name]
    interaction_coefs = params[interaction_coef_names]

    if main_effect_coef == 0 and len(interaction_coefs) == 0:
        return None, None, None

    spline_basis_pred = patsy.dmatrix(f"cr(year, df={spline_df})", {"year": year_range}, return_type='dataframe').iloc[:, 1:]

    marginal_effects, std_errors = [], []
    for i in range(len(year_range)):
        spline_vals_for_year = spline_basis_pred.iloc[i].values
        effect = main_effect_coef + np.dot(interaction_coefs, spline_vals_for_year)
        marginal_effects.append(effect)
        
        c = pd.Series(0.0, index=param_names)
        if weather_s_name in c.index: c[weather_s_name] = 1.0
        for coef_name, spline_val in zip(interaction_coef_names, spline_vals_for_year): c[coef_name] = spline_val
        var = c.T @ cov_matrix @ c
        std_errors.append(np.sqrt(var))

    marginal_effects = np.array(marginal_effects)
    std_errors = np.array(std_errors)
    ci_lower = marginal_effects - 1.96 * std_errors
    ci_upper = marginal_effects + 1.96 * std_errors

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(year_range, marginal_effects, color='black', lw=2.5, label=f'Time-Varying Effect (Spline)')
    ax.fill_between(year_range, ci_lower, ci_upper, color='grey', alpha=0.35, label='95% Confidence Interval')
    ax.axhline(0, color='red', linestyle='--', lw=1.5)
    ax.set_xlabel("Year", fontsize=14); ax.set_ylabel("Coefficient Value (Marginal Effect)", fontsize=14)
    ax.set_title(f"Spline Model: Evolution of Yield Sensitivity to {title_name}", fontsize=16, weight='bold')
    ax.legend(fontsize=12); ax.tick_params(axis='both', which='major', labelsize=12)
    
    plot_filename = os.path.join(output_path, f"panel_spline_beta_evolution_{weather_s_name}.png")
    fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show(); plt.close(fig)
    print(f"Saved spline coefficient plot to: {plot_filename}")
    
    return marginal_effects, ci_lower, ci_upper


def _plot_global_trend(results, year_range, output_path, title_name="Global Trend"):
    """Plots the coefficient for the Global Trend variable."""
    params = results.params
    sanitized_trend_name = 'Global_Trend'

    if sanitized_trend_name not in params.index:
        return None, None, None
        
    coef = params[sanitized_trend_name]
    std_err = results.bse[sanitized_trend_name]
    ci_lower, ci_upper = coef - 1.96 * std_err, coef + 1.96 * std_err
    
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axhline(coef, color='black', lw=2.5, label=f'Fixed Coefficient: {coef:.4f}')
    ax.axhspan(ci_lower, ci_upper, color='grey', alpha=0.35, label='95% Confidence Interval')
    ax.set_xlabel("Year", fontsize=14); ax.set_ylabel("Coefficient Value (Marginal Effect)", fontsize=14)
    ax.set_title(f"Spline Model: Estimated Coefficient for {title_name}", fontsize=16, weight='bold')
    ax.legend(fontsize=12); ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_ylim(ci_lower - (std_err * 2), ci_upper + (std_err * 2))
    
    plot_filename = os.path.join(output_path, "panel_spline_global_trend.png")
    fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show(); plt.close(fig)
    print(f"Saved Global Trend plot to: {plot_filename}")
    
    # Return as arrays for consistency
    return np.full_like(year_range, coef, dtype=float), np.full_like(year_range, ci_lower, dtype=float), np.full_like(year_range, ci_upper, dtype=float)



def _calculate_coefficients(results, spline_df, weather_s_name, year_range):
    """
    MANUAL CALCULATION of a single time-varying coefficient series and its confidence intervals.
    This function ONLY performs calculations and does not create plots.
    """
    params = results.params
    cov_matrix = results.cov_params()
    param_names = params.index

    main_effect_coef = params.get(weather_s_name, 0)
    interaction_coef_names = [name for name in param_names if f"*{weather_s_name}" in name]
    interaction_coefs = params[interaction_coef_names]

    # If the variable isn't in the model, return None
    if main_effect_coef == 0 and not interaction_coef_names:
        return None, None, None

    # Create the spline basis for the prediction range
    spline_basis_pred = patsy.dmatrix(f"cr(year, df={spline_df})", {"year": year_range}, return_type='dataframe').iloc[:, 1:]

    marginal_effects, std_errors = [], []
    for i in range(len(year_range)):
        spline_vals_for_year = spline_basis_pred.iloc[i].values
        # Calculate the marginal effect for the year
        effect = main_effect_coef + np.dot(interaction_coefs, spline_vals_for_year)
        marginal_effects.append(effect)
        
        # Calculate the variance for the marginal effect
        c = pd.Series(0.0, index=param_names)
        if weather_s_name in c.index: c[weather_s_name] = 1.0
        for coef_name, spline_val in zip(interaction_coef_names, spline_vals_for_year):
            c[coef_name] = spline_val
        var = c.T @ cov_matrix @ c
        std_errors.append(np.sqrt(var))

    marginal_effects = np.array(marginal_effects)
    std_errors = np.array(std_errors)
    ci_lower = marginal_effects - 1.96 * std_errors
    ci_upper = marginal_effects + 1.96 * std_errors
    
    return marginal_effects, ci_lower, ci_upper




def _plot_single_coefficient_with_ci(beta_t, ci_lower, ci_upper, year_range, title, output_path, filename):
    """Helper function to create and save a single coefficient plot with CI."""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(year_range, beta_t, color='black', lw=2.5, label='Time-Varying Effect (Spline)')
    ax.fill_between(year_range, ci_lower, ci_upper, color='grey', alpha=0.35, label='95% Confidence Interval')
    ax.axhline(0, color='red', linestyle='--', lw=1.5)
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel("Coefficient Value (Marginal Effect)", fontsize=14)
    ax.set_title(title, fontsize=16, weight='bold')
    ax.legend(fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    full_plot_path = os.path.join(output_path, filename)
    fig.savefig(full_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(f"Saved spline coefficient plot to: {full_plot_path}")


def plot_spline_results(all_results, spline_df_list, var_names, var_titles, years, output_path, run_note):
    """
    Orchestrates plotting based on the number of spline DFs run.
    - For a single DF, plots coefficients with confidence intervals.
    - For multiple DFs, plots a comparison of coefficient estimates.
    """
    print("\n" + "="*70)
    print("STARTING PLOT GENERATION")
    print("="*70)

    var_titles_dict = dict(zip(var_names, var_titles))
    # Assume all runs have the same year range, get it from the first result
    first_df = spline_df_list[0]
    year_range = np.arange(min(years), max(years) + 1)

    # CASE 1: Only one DF was run, plot with confidence intervals
    if len(spline_df_list) == 1:
        print(f"Plotting results for single DF={first_df} with confidence intervals...")
        results_for_df = all_results[first_df]
        for var in var_names:
            if var in results_for_df['plotting_results']:
                plot_data = results_for_df['plotting_results'][var]
                title = var_titles_dict.get(var, var)
                filename = f"plot_single_df_{run_note}_var_{var.replace(' ', '_')}.png"
                # Call the local helper function
                _plot_single_coefficient_with_ci(plot_data['beta_t'], plot_data['ci_lower'], plot_data['ci_upper'],
                                                 year_range, title, output_path, filename)

    # CASE 2: Multiple DFs were run, plot a comparison without CIs
    else:
        print("Plotting comparison of coefficients across multiple DFs...")
        for var in var_names:
            plt.style.use('seaborn-v0_8-whitegrid')
            fig, ax = plt.subplots(figsize=(12, 7))
            
            for df in spline_df_list:
                if var in all_results[df]['plotting_results']:
                    beta_t = all_results[df]['plotting_results'][var]['beta_t']
                    ax.plot(year_range, beta_t, lw=2, label=f'DF = {df}')

            ax.axhline(0, color='black', linestyle='--', lw=1.5)
            #ax.set_xlabel("Year", fontsize=14); ax.set_ylabel("Coefficient Value (Marginal Effect)", fontsize=14)
            title = var_titles_dict.get(var, var)
            ax.set_title(f"Comparison of Spline Fits for: {title}", fontsize="x-large")
            ax.legend(fontsize=12)
            plt.tick_params(axis='both', labelsize=15)
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax = plt.gca()
            ax.xaxis.set_major_locator(MultipleLocator(10))
            
            filename = f"plot_multi_df_{run_note}_var_{var.replace(' ', '_')}.png"
            full_plot_path = os.path.join(output_path, filename)
            fig.savefig(full_plot_path, dpi=300, bbox_inches='tight')
            plt.show()
            plt.close(fig)
            print(f"Saved multi-DF comparison plot to: {full_plot_path}")





# In Panel_Spline.py (REPLACE the old function with this)


def _create_mean_fit_plot(
    df_long, results, mY, years, regions, spline_df, output_path, run_note_str
    ):
    """
    Private helper to generate and save a single plot comparing actual vs. fitted mean yield.
    """
    print(f"  Generating Mean Fit Plot for DF={spline_df}...")
    
    # 1. Reshape fitted values using the internal df_long DataFrame
    df_long_with_fits = df_long.assign(fitted_values=results.fittedvalues)
    mY_fitted_panel = df_long_with_fits.pivot(index='year', columns='region', values='fitted_values')
    mY_fitted = mY_fitted_panel.reindex(index=years, columns=regions)

    # 2. Calculate actual and fitted means
    actual_mean_y = np.nanmean(mY, axis=1)
    fitted_mean_y = mY_fitted.mean(axis=1)

    # 3. Create the plot
    from matplotlib.ticker import MultipleLocator # Local import
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(years, actual_mean_y, '-', color='black', linewidth=2.5, label='Actual Mean Yield')
    ax.plot(years, fitted_mean_y, '--', linewidth=2, label=f'Fitted Mean (DF={spline_df})')
    
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.grid(linestyle='dashed')
    ax.set_ylabel('Cross-sectional Mean Log Yield', fontsize="x-large", labelpad=15)
    ax.tick_params(axis='both', labelsize=15)
    ax.legend(fontsize="large", loc='best')
    ax.set_title(f"Spline Model Fit (DF={spline_df})\n{run_note_str}", fontsize=16)
    plt.tight_layout()
    
    figures_path = os.path.join(output_path, "figures")
    os.makedirs(figures_path, exist_ok=True)
    plot_filename = os.path.join(figures_path, f"mean_fit_plot_{run_note_str}.png")
    fig.savefig(plot_filename, dpi=300)
    plt.show()
    plt.close(fig)
    print(f"  Saved mean fit plot to: {plot_filename}")



##############################
# MAIN:
#############################

def main_panel_spline(mY, mZ, years, regions, var_names, weather_vars, var_titles,
                      output_path, run_note_str, spline_df=5, get_fit_plot=False):
    """
    Main function for the Panel OLS model.
    This version focuses on ESTIMATION and CALCULATION, not plotting.
    """
    print("=" * 70)
    print(f"Running Panel Spline Estimation for: {run_note_str}")
    
    # --- 1. Data Preparation & Model Setup (Unchanged) ---
    df_long, name_map = _prepare_long_format_df(mY, mZ, years, regions, var_names)
    sanitized_weather_vars = [name_map[v] for v in weather_vars]
    sanitized_var_names = [name_map[v] for v in var_names]

    y = df_long['yield']
    X_fixed_effects = pd.get_dummies(df_long['region'], prefix='region', drop_first=True, dtype=float)
    control_vars_names = [name for name in sanitized_var_names if name not in sanitized_weather_vars]
    X_controls = df_long[control_vars_names]
    spline_basis = patsy.dmatrix(f"cr(year, df={spline_df})", data=df_long, return_type='dataframe')
    
    X_interactions_list = []
    for var in sanitized_weather_vars:
        main_effect = df_long[[var]]
        X_interactions_list.append(main_effect)
        interaction_terms = spline_basis.iloc[:, 1:].multiply(df_long[var], axis=0)
        interaction_terms.columns = [f"{c.replace(']', '_inter]')}*{var}" for c in interaction_terms.columns]
        X_interactions_list.append(interaction_terms)

    X = pd.concat([X_fixed_effects, X_controls] + X_interactions_list, axis=1)
    
    # --- 2. Run OLS Model (Unchanged) ---
    model = sm.OLS(y, X)
    results = model.fit(cov_type='cluster', cov_kwds={'groups': df_long['year']})

    summary_path = os.path.join(output_path, f"panel_spline_summary_{run_note_str}.txt")
    with open(summary_path, 'w') as f: f.write(results.summary().as_text())
    print(f"Saved model summary to: {summary_path}")

    # --- 3. NEW: Always CALCULATE coefficients for the results dictionary ---
    var_titles_dict = dict(zip(var_names, var_titles))
    year_range = np.arange(min(years), max(years) + 1)
    
    
    #NEW: FOR r2 PLOTTING
    df_long_with_fits = df_long.assign(fitted_values=results.fittedvalues)
    # Pivot the fitted values to get a (T, N) panel matrix
    mY_fitted_panel = df_long_with_fits.pivot(index='year', columns='region', values='fitted_values')
    # Ensure the columns and index match the original mY for consistency
    mY_fitted_panel = mY_fitted_panel.reindex(index=years, columns=regions)
    # -------
    
    plotting_results = {} # This will now hold the numerical results for plotting
    
    all_vars_to_calc = var_names
    for var in all_vars_to_calc:
        s_var = name_map[var]
        
        # We now always calculate the coefficients using our new helper function
        if var in weather_vars:
            beta, ci_l, ci_u = _calculate_coefficients(results, spline_df, s_var, year_range)
        else: # This handles any fixed-coefficient variables, if they exist
            # This logic can be simplified, but we keep it for now.
            # In our pipeline, this 'else' branch is currently not used.
            coef = results.params.get(s_var)
            if coef is not None:
                std_err = results.bse[s_var]
                beta = np.full_like(year_range, coef, dtype=float)
                ci_l = np.full_like(year_range, coef - 1.96 * std_err, dtype=float)
                ci_u = np.full_like(year_range, coef + 1.96 * std_err, dtype=float)
            else:
                beta, ci_l, ci_u = None, None, None

        if beta is not None:
            plotting_results[var] = {'beta_t': beta, 'ci_lower': ci_l, 'ci_upper': ci_u}
            

    if get_fit_plot:
        _create_mean_fit_plot(
            df_long=df_long,
            results=results,
            mY=mY,
            years=years,
            regions=regions,
            spline_df=spline_df,
            output_path=output_path,
            run_note_str=run_note_str
        ) 
            
    # --- 4. Package and Return Results ---
    results_dict = {'model_results': results, 'plotting_results': plotting_results, 'fitted_values_panel': mY_fitted_panel}
    
    return output_path, results_dict


# def main_panel_spline(mY, mZ, years, regions, var_names, weather_vars, var_titles,
#                       output_path, run_note_str, spline_df=5,
#                       # --- NEW PARAMETER ---
#                       make_plots=True):
#     """Main function for the Panel OLS model with time-varying spline coefficients."""
#     print("=" * 70)
#     print("Running Panel Regression (Manual Matrix) with Time-Varying Spline Coefficients")
#     print(f"Run Note: {run_note_str}, Spline DF: {spline_df}")
#     print("=" * 70)

#     # --- 1. Data Preparation ---
#     # Convert panel arrays to a long-format DataFrame for statsmodels
#     df_long, name_map = _prepare_long_format_df(mY, mZ, years, regions, var_names)
#     sanitized_weather_vars = [name_map[v] for v in weather_vars]
#     sanitized_var_names = [name_map[v] for v in var_names]

#     # --- 2. Construct the Design Matrix (X) and Dependent Variable (y) ---
#     y = df_long['yield']

#     # a) Fixed effects for each region
#     X_fixed_effects = pd.get_dummies(df_long['region'], prefix='region', drop_first=True, dtype=float)

#     # b) Control variables (regressors that do NOT get a time-varying coefficient)
#     control_vars_names = [name for name in sanitized_var_names if name not in sanitized_weather_vars]
#     X_controls = df_long[control_vars_names]

#     # c) Time-varying spline basis
#     spline_basis = patsy.dmatrix(f"cr(year, df={spline_df})", data=df_long, return_type='dataframe')

#     # d) Main effects and interaction terms for time-varying regressors
#     X_interactions_list = []
#     for var in sanitized_weather_vars:
#         # Add the main effect (e.g., the 'GDD' column itself)
#         main_effect = df_long[[var]]
#         X_interactions_list.append(main_effect)
#         # Create interactions by multiplying the variable with the spline basis columns
#         interaction_terms = spline_basis.iloc[:, 1:].multiply(df_long[var], axis=0)
#         interaction_terms.columns = [f"{c.replace(']', '_inter]')}*{var}" for c in interaction_terms.columns]
#         X_interactions_list.append(interaction_terms)

#     # e) Combine all parts into the final design matrix
#     X = pd.concat([X_fixed_effects, X_controls] + X_interactions_list, axis=1)

#     # --- 3. Run the OLS Model ---
#     # We cluster standard errors by year to account for time-based shocks
#     model = sm.OLS(y, X)
#     results = model.fit(cov_type='cluster', cov_kwds={'groups': df_long['year']})

#     # --- 4. Save Model Summary ---
#     summary_path = os.path.join(output_path, f"panel_spline_summary_{run_note_str}_df{spline_df}.txt")
#     with open(summary_path, 'w') as f: f.write(results.summary().as_text())
#     print(f"\nSaved model summary to: {summary_path}")

#     # --- 5. Generate and Save Plots (Conditionally) ---
#     var_titles_dict = dict(zip(var_names, var_titles))
#     year_range = np.arange(min(years), max(years) + 1)
#     plotting_results = {}

#     if make_plots:
#         print("\nGenerating and saving plots...")
#         all_vars_to_plot = var_names
#         for var in all_vars_to_plot:
#             s_var = name_map[var]
#             title = var_titles_dict.get(var, var)
            
#             # Check if the variable was modeled with a time-varying coefficient
#             if var in weather_vars:
#                 beta, ci_l, ci_u = _plot_time_varying_coefficient_manual(results, spline_df, s_var, year_range, output_path, title)
#             else:
#                 beta, ci_l, ci_u = _plot_global_trend(results, year_range, output_path, title_name=title)
                
#             if beta is not None:
#                 plotting_results[var] = {'beta_t': beta, 'ci_lower': ci_l, 'ci_upper': ci_u}
#     else:
#         print("\n`make_plots` is False, skipping plot generation.")
#         # We could calculate coefficients for the results dictionary here even if not plotting.
#         # For now, we keep it simple and skip this.

#     # --- 6. Package and Return Results ---
#     results_dict = {'model_results': results, 'plotting_results': plotting_results}
    
#     return output_path, results_dict