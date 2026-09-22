# Formerly: prepare_models_final_copy.py

import pandas as pd
import numpy as np
import os

# ---------------------------------------------------------------------------
# HELPER FUNCTIONS FOR BUILDING MODEL-SPECIFIC REGRESSOR MATRICES (mX)
# ---------------------------------------------------------------------------


###### MONHTLY DATA MODEL BUILDING:


def _build_model_0_vars(df_full_with_region_col, list_of_final_regions, season_ignored, config_opts):
    """
    Prepares the mX matrix for Model 0.
    This model has its own fixed logic: it uses 'temp_gs' and 'pcpn_gs', calculates
    anomalies against the full-period mean, and optionally adds squared terms.
    """
    print(f"Building variables for Model 0... (Season argument '{season_ignored}' is noted but Model 0 uses fixed GS variables).")

    temp_col_model0 = 'temp_gs' 
    pcpn_col_model0 = 'pcpn_gs' 

    df_model_vars = df_full_with_region_col[['year', 'region', temp_col_model0, pcpn_col_model0]].copy()

    # Calculate anomalies using full-period mean
    temp_anom_col_name = 'temp_gs_anom_model0_internal' 
    pcpn_anom_col_name = 'pcpn_gs_anom_model0_internal'
    df_model_vars[temp_anom_col_name] = df_model_vars[temp_col_model0] - df_model_vars.groupby('region')[temp_col_model0].transform('mean')
    df_model_vars[pcpn_anom_col_name] = df_model_vars[pcpn_col_model0] - df_model_vars.groupby('region')[pcpn_col_model0].transform('mean')

    # Pivot and impute
    df_temp_anom_pivoted = df_model_vars.pivot(index='year', columns='region', values=temp_anom_col_name)[list_of_final_regions].ffill().bfill()
    df_pcpn_anom_pivoted = df_model_vars.pivot(index='year', columns='region', values=pcpn_anom_col_name)[list_of_final_regions].ffill().bfill()

    regressors_to_stack_N_T = [df_temp_anom_pivoted.values.T, df_pcpn_anom_pivoted.values.T]
    
    var_names_list = ['Temp Anomaly (GS-M0)', 'Pcpn Anomaly (GS-M0)']
    ctrl_names_list = ['TempAnomGSM0', 'PcpnAnomGSM0']
    var_titles_list = ['Temperature Anomaly (GS - Model 0)', 'Precipitation Anomaly (GS - Model 0)']

    if config_opts.get('include_squares_model_0', True):
        regressors_to_stack_N_T.extend([(df_temp_anom_pivoted ** 2).values.T, (df_pcpn_anom_pivoted ** 2).values.T])
        var_names_list.extend(['Temp Anomaly² (GS-M0)', 'Pcpn Anomaly² (GS-M0)'])
        ctrl_names_list.extend(['TempAnomSqGSM0', 'PcpnAnomSqGSM0'])
        var_titles_list.extend(['Temperature Anomaly² (GS - Model 0)', 'Precipitation Anomaly² (GS - Model 0)'])
        
    mX = np.stack(regressors_to_stack_N_T, axis=2)
    return mX, var_names_list, ctrl_names_list, var_titles_list


def _build_model_a_vars(df_full_with_region_col, list_of_final_regions, season_ignored, config_opts):
    """
    Prepares the mX matrix for Model A ('temp_gs' and 'pcpn_gs').
    This version dynamically uses raw, detrended, or anomaly values,
    and includes squared terms based on config_opts.
    """
    print(f"Building variables for Model A... (Season argument '{season_ignored}' is noted but Model A uses fixed 'temp_gs' & 'pcpn_gs').")

    # --- 1. Determine which version of regressors to use ---
    processing_type = config_opts.get('regressor_processing', 'raw').lower()
    suffix = ''
    if processing_type == 'detrended':
        suffix = '_dtr'
    elif processing_type == 'anomaly':
        suffix = '_anom'
    
    print(f"  Using '{processing_type}' versions of regressors (suffix: '{suffix}').")

    # --- 2. Define base column names and construct final names ---
    temp_base_col = 'temp_gs'
    pcpn_base_col = 'pcpn_gs'
    
    temp_col_to_use = f'{temp_base_col}{suffix}'
    pcpn_col_to_use = f'{pcpn_base_col}{suffix}'

    # --- 3. Pivot linear terms and build initial regressor list ---
    pivoted_dfs_to_stack = []
    base_cols_to_pivot = [temp_col_to_use, pcpn_col_to_use]
    for col in base_cols_to_pivot:
        if col not in df_full_with_region_col.columns:
            raise ValueError(f"Required column '{col}' for Model A ({processing_type}) not found.")
        
        pivoted_df = df_full_with_region_col.pivot(index='year', columns='region', values=col)
        pivoted_df = pivoted_df[list_of_final_regions].ffill().bfill()
        pivoted_dfs_to_stack.append(pivoted_df)
        
    # --- 4. Handle squared terms based on square options AND processing type ---
    squares_option = config_opts.get('model_a_squares', 'none').lower()
    print(f"  Square option for Model A: '{squares_option}'")
    
    df_temp_pivoted = pivoted_dfs_to_stack[0]
    df_pcpn_pivoted = pivoted_dfs_to_stack[1]
    
    # Internal helper to get the correct squared term DataFrame
    def get_squared_df(base_pivoted_df, base_col_name, processing_type):
        if processing_type == 'raw':
            return base_pivoted_df ** 2
        else: 
            if processing_type == 'detrended':
                sq_col_name = f'{base_col_name}_sq_dtr'
            else: # anomaly
                sq_col_name = f'{base_col_name}_anom_sq'
            
            if sq_col_name not in df_full_with_region_col.columns:
                raise ValueError(f"Required squared column '{sq_col_name}' not found for Model A ({processing_type}).")
            
            pivoted_sq_df = df_full_with_region_col.pivot(index='year', columns='region', values=sq_col_name)[list_of_final_regions]
            return pivoted_sq_df.ffill().bfill()

    # Dynamically build the name lists as we add variables
    processing_label = f"({processing_type.capitalize()})" if suffix else "(Raw)"
    var_names_list = [f'Temp {processing_label} (GS)', f'Pcpn {processing_label} (GS)']
    ctrl_names_list = [f'temp_gs{suffix}', f'pcpn_gs{suffix}']
    var_titles_list = [f'Temperature {processing_label} (GS)', f'Precipitation {processing_label} (GS)']
    
    if squares_option == 'all':
        pivoted_dfs_to_stack.append(get_squared_df(df_temp_pivoted, temp_base_col, processing_type))
        var_names_list.append(f'Temp² {processing_label} (GS)')
        ctrl_names_list.append(f'temp_gs{suffix}_sq')
        var_titles_list.append(f'Temperature² {processing_label} (GS)')

    if squares_option in ['all', 'precip']:
        pivoted_dfs_to_stack.append(get_squared_df(df_pcpn_pivoted, pcpn_base_col, processing_type))
        var_names_list.append(f'Pcpn² {processing_label} (GS)')
        ctrl_names_list.append(f'pcpn_gs{suffix}_sq')
        var_titles_list.append(f'Precipitation² {processing_label} (GS)')

    # --- 5. Stack into mX and return ---
    regressors_to_stack_N_T = [df.values.T for df in pivoted_dfs_to_stack]
    mX = np.stack(regressors_to_stack_N_T, axis=2)

    return mX, var_names_list, ctrl_names_list, var_titles_list



###### DAILY DATA MODEL BUILDING:



def _get_squared_df(df_full, base_pivoted_df, base_col_name, processing_type, model_name, final_regions):
    """Internal helper to get the correct squared term DataFrame."""
    if processing_type == 'raw':
        return base_pivoted_df ** 2
    else: # detrended or anomaly
        if processing_type == 'detrended':
            sq_col_name = f'{base_col_name}_sq_dtr'
        else: # anomaly
            sq_col_name = f'{base_col_name}_anom_sq'
        
        if sq_col_name not in df_full.columns:
            raise ValueError(f"Required squared column '{sq_col_name}' not found for {model_name} ({processing_type}).")
        
        pivoted_sq_df = df_full.pivot(index='year', columns='region', values=sq_col_name)[final_regions]
        return pivoted_sq_df.ffill().bfill()

# In prepare_models_final.py

def _build_flexible_model_vars(df_full, final_regions, season, config_opts, model_info):
    """
    Generic template for building flexible models.
    This version now supports creating interaction terms.
    """
    print(f"Building variables for {model_info['name']}, Season: {season}...")
    processing_type = config_opts.get('regressor_processing', 'raw').lower()
    suffix = '_dtr' if processing_type == 'detrended' else '_anom' if processing_type == 'anomaly' else ''
    print(f"  Using '{processing_type}' versions of regressors (suffix: '{suffix}').")

    base_cols = [f"{var}_{season}" for var in model_info['vars']]
    cols_to_use = [f"{col}{suffix}" for col in base_cols]

    pivoted_dfs = {}
    for i, col in enumerate(cols_to_use):
        if col not in df_full.columns: raise ValueError(f"Required column '{col}' for {model_info['name']} not found.")
        pivoted_df = df_full.pivot(index='year', columns='region', values=col)[final_regions].ffill().bfill()
        pivoted_dfs[base_cols[i]] = pivoted_df

    regressors_to_stack = [pivoted_dfs[col] for col in base_cols]
    
    processing_label = f" ({processing_type.capitalize()})" if suffix else " (Raw)"
    name_map = model_info['name_map']
    var_names_list = [f'{v["name"]}{processing_label} ({season})' for k, v in name_map.items() if k in model_info['vars']]
    ctrl_names_list = [f'{v["ctrl"]}{season}{suffix}' for k, v in name_map.items() if k in model_info['vars']]
    var_titles_list = [f'{v["title"]}{processing_label} ({season})' for k, v in name_map.items() if k in model_info['vars']]

    # Handle squared terms (this logic is unchanged)
    squares_key = model_info.get('squares_key')
    squares_option = config_opts.get(squares_key, 'none').lower() if squares_key else model_info.get('default_squares', 'none')
    if squares_key: print(f"  Square option for {model_info['name']}: '{squares_option}'")
    
    if squares_option == 'all' and 'TMAX_AVG' in model_info['vars']:
        base_col = f'TMAX_AVG_{season}'
        regressors_to_stack.append(_get_squared_df(df_full, pivoted_dfs[base_col], base_col, processing_type, model_info['name'], final_regions))
        var_names_list.append(f'Mean Max Temp²{processing_label} ({season})')
        ctrl_names_list.append(f'TmaxSq{season}{suffix}')
        var_titles_list.append(f'Temperature²{processing_label} ({season})')

    if squares_option in ['all', 'precip'] or (squares_key is None and 'PREC' in model_info['vars']):
        base_col = f'PREC_{season}'
        regressors_to_stack.append(_get_squared_df(df_full, pivoted_dfs[base_col], base_col, processing_type, model_info['name'], final_regions))
        var_names_list.append(f'Total Precip²{processing_label} ({season})')
        ctrl_names_list.append(f'PrecSq{season}{suffix}')
        var_titles_list.append(f'Precipitation²{processing_label} ({season})')

    # --- ADD THIS NEW BLOCK TO CREATE INTERACTION TERMS ---
    interaction_pairs = model_info.get('interaction_terms', [])
    for pair in interaction_pairs:
        var1_key, var2_key = pair
        print(f"  Creating interaction term: {var1_key} * {var2_key}")

        base_col1 = f"{var1_key}_{season}"
        base_col2 = f"{var2_key}_{season}"

        # Get the already-pivoted DataFrames
        df1 = pivoted_dfs.get(base_col1)
        df2 = pivoted_dfs.get(base_col2)

        if df1 is not None and df2 is not None:
            # Create the interaction via element-wise multiplication
            interaction_df = df1 * df2
            regressors_to_stack.append(interaction_df)

            # Create names for the new interaction variable
            var_names_list.append(f'{name_map[var1_key]["ctrl"]}x{name_map[var2_key]["ctrl"]}{processing_label} ({season})')
            ctrl_names_list.append(f'{name_map[var1_key]["ctrl"]}x{name_map[var2_key]["ctrl"]}{season}{suffix}')
            var_titles_list.append(f'Interaction: {name_map[var1_key]["title"]} x {name_map[var2_key]["title"]}{processing_label} ({season})')
        else:
            print(f"Warning: Could not create interaction for {pair}. One or both base variables not found.")
    # --- END OF NEW BLOCK ---

    # # --- THIS IS THE UPDATED BLOCK for interaction terms ---
    # interaction_pairs = model_info.get('interaction_terms', [])
    # for pair in interaction_pairs:
    #     var1_key, var2_key = pair
    #     print(f"  Creating standardized interaction term: {var1_key} * {var2_key}")

    #     base_col1 = f"{var1_key}_{season}"
    #     base_col2 = f"{var2_key}_{season}"

    #     # Get the already-pivoted DataFrames
    #     df1 = pivoted_dfs.get(base_col1)
    #     df2 = pivoted_dfs.get(base_col2)

    #     if df1 is not None and df2 is not None:
    #         # --- NEW: Standardize each variable before interacting ---
    #         # Standardize by subtracting the mean and dividing by the standard deviation
    #         # of the entire panel data for that variable.
    #         df1_std = (df1 - df1.mean().mean()) / df1.stack().std()
    #         df2_std = (df2 - df2.mean().mean()) / df2.stack().std()
            
    #         # Create the interaction term by multiplying the STANDARDIZED variables
    #         interaction_df = df1_std * df2_std
    #         regressors_to_stack.append(interaction_df)

    #         # Add corresponding names and titles for the new variable
    #         # Adding "Std" to the names to make it clear what they represent
    #         var_names_list.append(f'{name_map[var1_key]["ctrl"]}x{name_map[var2_key]["ctrl"]} (Std){processing_label} ({season})')
    #         ctrl_names_list.append(f'{name_map[var1_key]["ctrl"]}x{name_map[var2_key]["ctrl"]}Std{season}{suffix}')
    #         var_titles_list.append(f'Interaction: Std. {name_map[var1_key]["title"]} x Std. {name_map[var2_key]["title"]}{processing_label} ({season})')
    #     else:
    #         print(f"Warning: Could not create interaction for {pair}. One or both base variables not found.")
    # # --- END OF UPDATED BLOCK ---

    # Final stack and return
    regressors_to_stack_N_T = [df.values.T for df in regressors_to_stack]
    mX = np.stack(regressors_to_stack_N_T, axis=2)
    return mX, var_names_list, ctrl_names_list, var_titles_list



def _build_model_1_vars(df, regions, season, config):
    model_info = {
        'name': 'Model 1', 'vars': ['TMAX_AVG', 'PREC'], 'squares_key': 'model_1_squares',
        'name_map': {'TMAX_AVG': {'name':'Mean Max Temp','ctrl':'Tmax','title':'Temperature'}, 
                     'PREC': {'name':'Total Precip','ctrl':'Prec','title':'Precipitation'}}
    }
    return _build_flexible_model_vars(df, regions, season, config, model_info)

def _build_model_2_vars(df, regions, season, config):
    model_info = {
        'name': 'Model 2', 'vars': ['GDD', 'KDD', 'PREC'], 'default_squares': 'precip',
        'name_map': {'GDD': {'name':'GDD','ctrl':'GDD','title':'GDD'}, 
                     'KDD': {'name':'KDD','ctrl':'KDD','title':'KDD'}, 
                     'PREC': {'name':'Precip','ctrl':'Prec','title':'Precipitation'}}
    }
    return _build_flexible_model_vars(df, regions, season, config, model_info)

# In prepare_models_final.py

def _build_model_3_vars(df, regions, season, config):
    model_info = {
        'name': 'Model 3',
        # We now only need the base variables for the interaction
        'vars': ['GDD', 'KDD', 'PREC'], 
        'default_squares': 'precip',
        # --- NEW: Define the interaction term we want to create ---
        'interaction_terms': [('KDD', 'PREC')],
        
        'name_map': {
            'GDD': {'name':'GDD','ctrl':'GDD','title':'GDD'}, 
            'KDD': {'name':'KDD','ctrl':'KDD','title':'KDD'}, 
            'PREC': {'name':'Precip','ctrl':'Prec','title':'Precipitation'}
            # CHD is removed from the name map
        }
    }
    return _build_flexible_model_vars(df, regions, season, config, model_info)


DAILY_VARS_NAME_MAP = {
    'GDD': {'name':'GDD','ctrl':'GDD','title':'GDD'}, 
    'KDD': {'name':'KDD','ctrl':'KDD','title':'KDD'}, 
    'PREC': {'name':'Total Precip','ctrl':'Prec','title':'Precipitation'},
    'TMAX_AVG': {'name':'Mean Max Temp','ctrl':'Tmax','title':'Temperature'},
    'CDD_1mm': {'name':'CDD (<1mm)','ctrl':'CDD1mm','title':'Consecutive Dry Days (<1mm)'},
    'CDD_2mm': {'name':'CDD (<2mm)','ctrl':'CDD2mm','title':'Consecutive Dry Days (<2mm)'},
    'Rx5day': {'name':'Max 5-Day Rain','ctrl':'Rx5day','title':'Max 5-Day Precip'},
    'R10mm': {'name':'Heavy Rain Days (>10mm)','ctrl':'R10mm','title':'Heavy Rain Days (>10mm)'},
    'R20mm': {'name':'Heavy Rain Days (>20mm)','ctrl':'R20mm','title':'Heavy Rain Days (>20mm)'}
}

def _build_model_4_vars(df, regions, season, config):
    """Model 4: GDD, KDD, CDD, and Rx5day for the Growing Season."""
    model_info = {
        'name': 'Model 4',
        'vars': ['GDD', 'KDD', 'CDD_2mm', 'Rx5day'],
        'default_squares': 'none',
        'name_map': DAILY_VARS_NAME_MAP
    }
    # Per your request, this model always uses the 'GS' season
    return _build_flexible_model_vars(df, regions, season='GS', config_opts=config, model_info=model_info)

def _build_model_5_vars(df, regions, season, config):
    """Model 5: GDD, KDD, CDD, and R20mm for the Growing Season."""
    model_info = {
        'name': 'Model 5',
        'vars': ['GDD', 'KDD', 'CDD_2mm', 'R20mm'],
        'default_squares': 'none',
        'name_map': DAILY_VARS_NAME_MAP
    }
    # Per your request, this model always uses the 'GS' season
    return _build_flexible_model_vars(df, regions, season='GS', config_opts=config, model_info=model_info)


# In prepare_models_final_copy.py

# REPLACE the old _build_model_6_vars with this one:

def _build_model_6_vars(df, regions, season, config):
    """
    Model 6 (Corrected): GDD & KDD from GS, with simple quadratic precip from SGF.
    This is a custom builder because it mixes seasons.
    """
    print(f"Building variables for Model 6 (Mixed Season)...")
    processing_type = config.get('regressor_processing', 'raw').lower()
    suffix = '_dtr' if processing_type == 'detrended' else '_anom' if processing_type == 'anomaly' else ''
    print(f"  Using '{processing_type}' versions of regressors (suffix: '{suffix}').")

    # --- 1. Manually define the exact columns needed from different seasons ---
    cols_to_use = [
        f'GDD_GS{suffix}',
        f'KDD_GS{suffix}',
        f'PREC_SGF{suffix}'
    ]

    # --- 2. Pivot the linear terms ---
    pivoted_dfs_to_stack = []
    for col in cols_to_use:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' for Model 6 not found.")
        pivoted_df = df.pivot(index='year', columns='region', values=col)[regions].ffill().bfill()
        pivoted_dfs_to_stack.append(pivoted_df)
    
    df_prec_sgf_pivoted = pivoted_dfs_to_stack[2] # This is the PREC_SGF dataframe

    # --- 3. Add the squared precipitation term for SGF ---
    # The _get_squared_df helper function is perfect for this
    prec_sq_df = _get_squared_df(df, df_prec_sgf_pivoted, 'PREC_SGF', processing_type, 'Model 6', regions)
    pivoted_dfs_to_stack.append(prec_sq_df)

    # --- 4. Manually define the names to match the mixed seasons ---
    processing_label = f" ({processing_type.capitalize()})" if suffix else " (Raw)"
    var_names_list = [
        f'GDD{processing_label} (GS)',
        f'KDD{processing_label} (GS)',
        f'Total Precip{processing_label} (SGF)',
        f'Total Precip²{processing_label} (SGF)'
    ]
    ctrl_names_list = [
        f'GDDGS{suffix}',
        f'KDDGS{suffix}',
        f'PrecSGF{suffix}',
        f'PrecSqSGF{suffix}'
    ]
    var_titles_list = [
        f'GDD{processing_label} (GS)',
        f'KDD{processing_label} (GS)',
        f'Precipitation{processing_label} (SGF)',
        f'Precipitation²{processing_label} (SGF)'
    ]

    # --- 5. Stack into mX and return ---
    regressors_to_stack_N_T = [df.values.T for df in pivoted_dfs_to_stack]
    mX = np.stack(regressors_to_stack_N_T, axis=2)
    
    return mX, var_names_list, ctrl_names_list, var_titles_list


def _build_model_7_vars(df, regions, season, config):
    """Model 7: Compound Heat & Drought Stress (based on Model 4)."""
    model_info = {
        'name': 'Model 7',
        # Base variables are from Model 4
        'vars': ['GDD', 'KDD', 'CDD_2mm', 'R20mm'],
        'default_squares': 'none',
        'name_map': DAILY_VARS_NAME_MAP,
        
        # --- NEW: Define the interaction term we want to create ---
        'interaction_terms': [('KDD', 'CDD_2mm')]
    }
    # This model uses the 'GS' season
    return _build_flexible_model_vars(df, regions, season='GS', config_opts=config, model_info=model_info)


# In prepare_models_final_copy.py

# ... (after the _build_model_6_vars function) ...
# In prepare_models_final_copy.py

# REPLACE the _build_model_6b_vars with this corrected version:

def _build_model_6b_vars(df, regions, season, config):
    """
    Model 6_b (Corrected): GDD(GS) & KDD(GS), with quadratic PREC(SGF) and interaction KDD(GS) * PREC(SGF).
    Structurally identical to Model 6 + interaction.
    """
    print(f"Building variables for Model 6_b (Mixed Season + Interaction)...")
    processing_type = config.get('regressor_processing', 'raw').lower()
    suffix = '_dtr' if processing_type == 'detrended' else '_anom' if processing_type == 'anomaly' else ''
    print(f"  Using '{processing_type}' versions of regressors (suffix: '{suffix}').")

    # --- 1. Manually define the exact columns needed from different seasons ---
    cols_to_use = [
        f'GDD_GS{suffix}',
        f'KDD_GS{suffix}',
        f'PREC_SGF{suffix}'
    ]

    # --- 2. Pivot the linear terms AND store KDD_GS ---
    pivoted_dfs_to_stack = []
    df_kdd_gs_pivoted = None # Initialize to store KDD_GS
    df_prec_sgf_pivoted = None # Initialize to store PREC_SGF

    for col in cols_to_use:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' for Model 6_b not found.")
        pivoted_df = df.pivot(index='year', columns='region', values=col)[regions].ffill().bfill()
        pivoted_dfs_to_stack.append(pivoted_df)
        
        # Store the specific pivoted DataFrames needed for interaction
        if col == f'KDD_GS{suffix}':
            df_kdd_gs_pivoted = pivoted_df
        if col == f'PREC_SGF{suffix}':
            df_prec_sgf_pivoted = pivoted_df

    # --- 3. Add the squared precipitation term for SGF ---
    prec_sq_df = _get_squared_df(df, df_prec_sgf_pivoted, 'PREC_SGF', processing_type, 'Model 6_b', regions)
    pivoted_dfs_to_stack.append(prec_sq_df)

    # --- 4. Add the KDD_GS * PREC_SGF interaction term ---
    if df_kdd_gs_pivoted is None or df_prec_sgf_pivoted is None:
        raise ValueError("Could not find pivoted KDD_GS or PREC_SGF for interaction in Model 6_b.")
    interaction_df = df_kdd_gs_pivoted * df_prec_sgf_pivoted
    pivoted_dfs_to_stack.append(interaction_df) # Add interaction to the stack

    # --- 5. Manually define the names to match the mixed seasons AND interaction ---
    processing_label = f" ({processing_type.capitalize()})" if suffix else " (Raw)"
    var_names_list = [
        f'GDD{processing_label} (GS)',
        f'KDD{processing_label} (GS)',
        f'Total Precip{processing_label} (SGF)',
        f'Total Precip²{processing_label} (SGF)',
        f'KDDxPrec{processing_label} (GSxSGF)' # Interaction name
    ]
    ctrl_names_list = [
        f'GDDGS{suffix}',
        f'KDDGS{suffix}',
        f'PrecSGF{suffix}',
        f'PrecSqSGF{suffix}',
        f'KDDGSxPrecSGF{suffix}' # Interaction control name
    ]
    var_titles_list = [
        f'GDD{processing_label} (GS)',
        f'KDD{processing_label} (GS)',
        f'Precipitation{processing_label} (SGF)',
        f'Precipitation²{processing_label} (SGF)',
        f'Interaction: KDD(GS) x Prec(SGF){processing_label}' # Interaction title
    ]

    # --- 6. Stack into mX and return ---
    regressors_to_stack_N_T = [df.values.T for df in pivoted_dfs_to_stack]
    mX = np.stack(regressors_to_stack_N_T, axis=2)
    
    return mX, var_names_list, ctrl_names_list, var_titles_list


# ---------------------------------------------------------------------------
# MAIN DATA PREPARATION FUNCTION (DISPATCHER)
# ---------------------------------------------------------------------------

def model_specification(
    df_raw_input, 
    model_type, 
    season, 
    config_options=None 
    ):
    """
    Prepares LLDVE model inputs (mY, mX, mZ, names, etc.) based on specified
    model type, season, and processing options in config_options.
    """
    if config_options is None:
        config_options = {}

    df = df_raw_input.copy()

    # --- 1. Filter Data by State and Year ---
    states_to_include = config_options.get('states_to_include', None)
    if states_to_include:
        df['state'] = df['state'].astype(str).str.zfill(2) 
        states_to_include_str = [str(s).zfill(2) for s in states_to_include]
        df = df[df['state'].isin(states_to_include_str)].copy()
        if df.empty: raise ValueError(f"DataFrame empty after filtering for states: {states_to_include_str}")
    
    if model_type.lower() in ["model_1", "model_2", "model_3", "model_4", "model_5", "model_6", "model_6b", "model_7"]:
        effective_start_year = 1951
    else:
        effective_start_year = df['year'].min() if not df.empty else 1926 
        
    print(effective_start_year)
    
    df = df[df['year'] >= effective_start_year].copy()
    if df.empty: raise ValueError(f"DataFrame empty after year filtering (>= {effective_start_year}).")

    # --- 2. Common Data Transformations ---
    df['region'] = df['state'].astype(str).str.zfill(2) + '_' + df['county'].astype(str).str.zfill(3)
    
    # --- ADD THIS DIAGNOSTIC BLOCK TO FIND DUPLICATES ---
    key_cols = ['year', 'region']
    duplicates = df[df.duplicated(subset=key_cols, keep=False)]

    if not duplicates.empty:
        print("\n" + "="*50)
        print("!!! ERROR: Found duplicate entries for (year, region) pairs. Cannot pivot. !!!")
        print(f"Found {len(duplicates)} total rows that are part of a duplicate set.")
        print("Showing the first 10 problematic rows, sorted for comparison:")
        print(duplicates.sort_values(by=key_cols).head(10))
        print("="*50 + "\n")
        # You might want to stop the script here to investigate
        # raise ValueError("Stopping due to duplicate entries found.")
    # --- END OF DIAGNOSTIC BLOCK ---

    # --- 3. DYNAMICALLY SELECT AND PIVOT Y (Yield) ---
    yield_opts = config_options.get('yield_processing', {'log': True, 'detrend': False})
    use_log = yield_opts.get('log', True)
    use_detrend = yield_opts.get('detrend', False)
    
    if use_log and use_detrend: yield_col_to_use = 'log_value_dtr'
    elif use_log and not use_detrend: yield_col_to_use = 'log_value'
    elif not use_log and use_detrend: yield_col_to_use = 'value_dtr'
    else: yield_col_to_use = 'value'
        
    print(f"Using '{yield_col_to_use}' as the dependent variable (Y) for this run.")
    if yield_col_to_use not in df.columns:
        raise ValueError(f"Required yield column '{yield_col_to_use}' not found. Ensure pre-processing scripts have run.")
        
    df_yield_full = df.pivot(index='year', columns='region', values=yield_col_to_use).sort_index()

    # --- 4. Drop Bad Regions by Name ---
    region_names_to_drop = config_options.get('drop_region_names', [])
    if region_names_to_drop:
        valid_names_to_drop = [name for name in region_names_to_drop if name in df_yield_full.columns]
        df_yield = df_yield_full.drop(columns=valid_names_to_drop, errors='ignore')
    else:
        df_yield = df_yield_full
    
    # --- 5. Extract Final mY, years, regions ---
    years   = df_yield.index.to_numpy()
    regions = df_yield.columns.to_list()
    mY      = df_yield.values
    
    if mY.size == 0: raise ValueError(f"mY is empty after all filtering for {model_type}, {season}. Check filters.")
    print(f"Data prepared for {len(regions)} regions and {len(years)} years.")

    # --- 6. Model-Specific Regressor Preparation (mX) ---
    model_dispatcher = {
        "model_0": _build_model_0_vars,
        "model_a": _build_model_a_vars,
        "model_1": _build_model_1_vars,
        "model_2": _build_model_2_vars,
        "model_3": _build_model_3_vars,
        "model_4": _build_model_4_vars, 
        "model_5": _build_model_5_vars,
        "model_6": _build_model_6_vars, 
        "model_6b": _build_model_6b_vars, # <-- added
        "model_7": _build_model_7_vars  
    }
    build_func = model_dispatcher.get(model_type.lower())
    if not build_func:
        raise ValueError(f"Unknown model_type: '{model_type}'.")
        
    mX, model_var_names, model_ctrl_names, model_var_titles = \
        build_func(df, regions, season, config_options)

    # --- 7. Final Checks, mZ Construction, and Return ---
    if mX.shape[1] != len(years):
        raise ValueError(f"Time dimension mismatch for {model_type} {season}: mX has {mX.shape[1]} periods, mY has {len(years)}.")
        
    N_final, T_final, d_model_specific = mX.shape 
    mZ = np.ones((N_final, T_final, d_model_specific + 1))
    mZ[:,:,1:] = mX
    
    final_var_names = ['Global Trend'] + model_var_names
    final_ctrl_names = ['GT'] + model_ctrl_names
    final_var_titles = ['Intercept (Global Trend)'] + model_var_titles

    if not (mZ.shape[2] == len(final_var_names)):
        raise ValueError(f"Mismatch after adding Global Trend: Num regressors in mZ ({mZ.shape[2]}) vs var_names ({len(final_var_names)}).")
    
    return mY, mX, mZ, years, regions, final_var_names, final_ctrl_names, final_var_titles