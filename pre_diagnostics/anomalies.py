import os
import pandas as pd
import numpy as np
import re

def run_anomaly_calculation():
    """
    Main function to load the importer data, calculate anomalies for weather variables
    based on a fixed baseline for each county, and save the result to a master file.
    """
    print("--- Starting Anomaly Calculation Process ---")

    # --- 1. USER SETTINGS ---
    # Choose the method for calculating the squared anomaly term.
    # Options:
    #   1. 'square_first_then_subtract_mean': (Correct for quadratic models)
    #      Calculates anomaly of the squared term -> (PREC²) - mean(PREC²)
    #   2. 'subtract_mean_first_then_square': (Measures variance)
    #      Calculates squared anomaly -> (PREC - mean(PREC))²
    SQUARING_METHOD = 'subtract_mean_first_then_square'

    # Define the baseline period for calculating "normal" weather
    BASELINE_PERIOD = range(1981, 2011)  # Baseline is 1981-2010 inclusive

    # --- 2. PATH SETUP ---
    try:
        script_dir = os.path.dirname(__file__)
    except NameError:
        script_dir = os.getcwd()  # Fallback for interactive environments

    parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir)) #repo root, the master panels live there
    # Input is the output of missing_data.py (county filter applied), in this folder.
    input_filename = "df_final_dropped_corn.csv"#"df_final_dropped_soy.csv"#"df_final_dropped_east100m.csv"
    output_filename = "master_panel_corn.csv"#"master_panel_soy.csv"#"master_panel_corn_east100m.csv"
    input_path = os.path.join(script_dir, input_filename)
    output_path = os.path.join(parent_dir, output_filename) #written one level up

    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        print("Please run missing_data.py first.")
        return

    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)

    # --- 3. VALIDATE SETTINGS & IDENTIFY VARIABLES ---
    valid_methods = ['square_first_then_subtract_mean', 'subtract_mean_first_then_square']
    if SQUARING_METHOD not in valid_methods:
        raise ValueError(f"Error: SQUARING_METHOD must be one of {valid_methods}")
    print(f"\nUsing squaring method: '{SQUARING_METHOD}'")

    # The importer writes fips_full as '17001', the rest of the pipeline (and the
    # region names in the run notebooks) expects '17_001', so rebuild it here.
    # This used to happen in detrending.py, which is no longer part of the pipeline.
    df['fips_full'] = df['state'].astype(str).str.zfill(2) + '_' + \
                      df['county'].astype(str).str.zfill(3)

    # Log yield is the dependent variable of the paper models. Also used to live
    # in detrending.py.
    if 'log_value' not in df.columns and 'value' in df.columns:
        print("Creating 'log_value' column from 'value'.")
        df['log_value'] = np.log(df['value'])


    bin_cols = [
        c for c in df.columns
        if re.fullmatch(r"bin_-?\d+_(?:GS|SGF)", c)
    ]

    weather_vars = [
        # Original Monthly Variables (for Model 0, Model A)
        'temp_gs', 'pcpn_gs',
        # Daily Aggregate Variables (for Models 1, 2, 3, 6)
        'GDD_GS', 'KDD_GS', 'TMAX_AVG_GS', 'PREC_GS',
        'GDD_SGF', 'KDD_SGF', 'TMAX_AVG_SGF', 'PREC_SGF',
        # Concurrent hot-dry days, not used by a model yet but kept available
        'CHD_GS', 'CHD_SGF',
        # New Advanced Precipitation Metrics (for Models 4, 5)
        'CDD_1mm_GS', 'CDD_2mm_GS', 'Rx5day_GS', 'R10mm_GS', 'R20mm_GS',
        'CDD_1mm_SGF', 'CDD_2mm_SGF', 'Rx5day_SGF', 'R10mm_SGF', 'R20mm_SGF'
    ]
    
    # --- FINAL CORRECTED LIST: Includes all variables that might be squared ---
    vars_to_square = [
        'temp_gs', 'pcpn_gs',
        'TMAX_AVG_GS', 'TMAX_AVG_SGF',
        'PREC_GS', 'PREC_SGF'
    ]
    
    existing_weather_vars = [var for var in weather_vars if var in df.columns]
    print(f"Found {len(existing_weather_vars)} weather variables to process for anomalies.\n")

    # --- 4. CALCULATE NORMALS AND ANOMALIES ---
    df_baseline = df[df['year'].isin(BASELINE_PERIOD)]
    if df_baseline.empty:
        raise ValueError(f"No data found for the baseline period: {BASELINE_PERIOD}. Cannot calculate normals.")

    for var in existing_weather_vars:
        norm_col = f'{var}_norm'
        anom_col = f'{var}_anom'
        
        print(f"Processing: {var} -> {anom_col}")

        # 1. Compute normals (mean over baseline) for each county
        county_normals = df_baseline.groupby('fips_full')[var].mean()

        # 2. Map these normals back into the main DataFrame
        df[norm_col] = df['fips_full'].map(county_normals)

        # 3. Compute linear anomaly
        df[anom_col] = df[var] - df[norm_col]

        # 4. Create squared anomaly if required, using the selected method
        if var in vars_to_square:
            sq_anom_col = f'{var}_anom_sq'
            print(f"         ... also creating squared anomaly -> {sq_anom_col}")
            
            if SQUARING_METHOD == 'subtract_mean_first_then_square':
                # This method calculates (X - mean(X))²
                df[sq_anom_col] = df[anom_col] ** 2
            
            elif SQUARING_METHOD == 'square_first_then_subtract_mean':
                # This method calculates (X²) - mean(X²)
                sq_var = f'{var}_sq'
                norm_sq_col = f'{sq_var}_norm'
                
                # a. Square the variable
                df[sq_var] = df[var] ** 2
                
                # b. Calculate the mean of the *squared* variable over the baseline
                county_normals_sq = df_baseline.groupby('fips_full')[var].apply(lambda x: (x**2).mean())
                
                # c. Map the normal of the squared variable back
                df[norm_sq_col] = df['fips_full'].map(county_normals_sq)
                
                # d. Calculate the final anomaly
                df[sq_anom_col] = df[sq_var] - df[norm_sq_col]
                
                # e. Clean up temporary columns
                df = df.drop(columns=[sq_var, norm_sq_col])
        
        # Clean up the intermediate 'norm' column for the linear term
        df = df.drop(columns=[norm_col])

    # 5. --- SAVE FINAL MASTER DATASET ---
    print(f"\nSaving master data with raw and anomaly columns to: {output_path}")
    try:
        df.to_csv(output_path, index=False)
        print("Anomaly calculation complete. Master data file saved successfully.")
    except Exception as e:
        print(f"Error saving file: {e}")

    # --- Log what was made (panel_log.json). The filter settings come from the latest
    #     missing_data entry for our input file, so the run notebooks can check them. ---
    import json
    from datetime import datetime
    log_path = os.path.join(script_dir, "panel_log.json")
    log = json.load(open(log_path)) if os.path.exists(log_path) else []
    #the log is in build order, so the last matching entry is the file we just read
    last_filter_entry = None
    for entry in log:
        if entry.get('step') == 'missing_data' and entry.get('output_file') == input_filename:
            last_filter_entry = entry

    filter_info = {}
    if last_filter_entry is None:
        print(f"Warning: no missing_data entry for {input_filename} in panel_log.json; run missing_data.py first so the window is known.")
    else:
        for key in ('start_year', 'end_year', 'missing_year_tolerance', 'counties_in', 'counties_kept', 'created'):
            filter_info[key] = last_filter_entry.get(key)
    log.append({
        'created': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'step': 'anomalies',
        'output_file': output_filename,
        'input_file': input_filename,
        'baseline': f"{BASELINE_PERIOD.start}-{BASELINE_PERIOD.stop - 1}",
        'squaring_method': SQUARING_METHOD,
        'rows': len(df),
        'counties': df['fips_full'].nunique(),
        'filter': filter_info,   # copied from the missing_data entry, 'created' says which run
    })
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"Logged in: {log_path}")

if __name__ == '__main__':
    run_anomaly_calculation()