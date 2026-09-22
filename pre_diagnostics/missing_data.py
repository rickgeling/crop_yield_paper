import pandas as pd
import numpy as np
import os
import sys
import re

# --- USER SETTINGS: which dataset to filter ---
INPUT_FILENAME = "df_final_importer_corn_paper.csv"#"df_final_importer_soy_paper.csv"#"df_final_importer_corn_east100m.csv"
OUTPUT_FILENAME = "df_final_dropped_corn.csv"#"df_final_dropped_soy.csv"#"df_final_dropped_east100m.csv"

# --- USER SETTINGS: the filter window ---
START_YEAR = 1951 #nClimGrid-Daily starts in 1951, so weather is empty before that
END_YEAR = 2024 #last year that counts; None = everything in the file. Must match END_YEAR in the lldve_run notebooks (they check)
MISSING_YEAR_TOLERANCE = 4 #drop counties with 5 or more missing years in the window (Zipper et al. 2016 rule)

def get_base_path():
    """Determines the script's directory for robust path handling."""
    try:
        return os.path.dirname(__file__)
    except NameError:
        return os.getcwd()

def check_missing_data_by_column(start_year=1951):
    """
    Checks each key column individually for any missing data from a given start year
    on the original raw dataset.
    """
    print("--- Starting Column-wise Missing Data Check (1951 onwards on Raw Data) ---")
    
    script_dir = get_base_path()
    input_path = os.path.join(script_dir, INPUT_FILENAME) #set at the top of this file

    if not os.path.exists(input_path):
        print(f"Error: Input file not found at: {input_path}")
        return

    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    
    df = df[df['year'] >= start_year].copy()
    
    if df.empty:
        print(f"Error: No data available from {start_year} onwards. Stopping check.")
        return
    
    # --- CHANGED: Updated the list of key columns to check ---
    key_columns = [
        'value', #yield
        'GDD_GS', 'KDD_GS', 'PREC_GS',
        'CDD_1mm_GS', 'CDD_2mm_GS', 'Rx5day_GS', 'R10mm_GS', 'R20mm_GS', #precip extremes, only in the extremes importer files
        'GDD_SGF', 'KDD_SGF', 'PREC_SGF',
        'CDD_1mm_SGF', 'CDD_2mm_SGF', 'Rx5day_SGF', 'R10mm_SGF', 'R20mm_SGF'
    ]
    key_columns = [col for col in key_columns if col in df.columns] #columns that are not in this dataset are simply skipped

    print(f"\nChecking for any missing data in columns since {start_year}:")
    any_missing_found = False
    for col in key_columns:
        if df[col].isnull().any():
            missing_count = df[col].isnull().sum()
            print(f" - Column '{col}': YES, missing data found ({missing_count} rows).")
            any_missing_found = True
        else:
            print(f" - Column '{col}': NO, this column is complete.")
            
    if not any_missing_found:
        print("\nResult: No missing data was found in any of the key columns.")
    
    print("--- Column-wise check complete. ---\n")

def check_missing_data_full_timeseries(df, start_year=1951, end_year=None):
    """
    Checks a given DataFrame for counties with > 4 missing years and reports which years are missing.
    """
    print(f"--- Starting Full Timeseries Check on FINAL Filtered Data ({start_year}-{end_year if end_year is not None else 'end'}) ---")
    #same window as the filter, otherwise every county looks bad: there is no weather data before 1951
    df_check = df[df['year'] >= start_year].copy()
    if end_year is not None:
        df_check = df_check[df_check['year'] <= end_year]

    # --- CHANGED: Use the same comprehensive list of key columns ---
    key_columns = [
        'value', #yield
        'GDD_GS', 'KDD_GS', 'PREC_GS',
        'CDD_1mm_GS', 'CDD_2mm_GS', 'Rx5day_GS', 'R10mm_GS', 'R20mm_GS', #precip extremes, only in the extremes importer files
        'GDD_SGF', 'KDD_SGF', 'PREC_SGF',
        'CDD_1mm_SGF', 'CDD_2mm_SGF', 'Rx5day_SGF', 'R10mm_SGF', 'R20mm_SGF'
    ]
    key_columns = [col for col in key_columns if col in df_check.columns]

    #a year counts as missing when ANY key column is empty in that county-year
    df_check['is_missing_year'] = df_check[key_columns].isnull().any(axis=1)
    missing_year_counts = df_check.groupby('fips_full')['is_missing_year'].sum()

    counties_to_report = missing_year_counts[missing_year_counts > 4].index

    if not counties_to_report.empty:
        print(f"\nFound {len(counties_to_report)} counties with more than 4 missing years in the FINAL dataset:")
        for fips in counties_to_report:
            num_missing = missing_year_counts[fips]
            county_data = df_check[df_check['fips_full'] == fips]
            missing_years = county_data[county_data['is_missing_year']]['year'].tolist()
            print(f"  - County {fips} has {int(num_missing)} total missing years: {missing_years}")
    else:
        print("\nNo counties found with more than 4 missing years in the FINAL dataset.")

    print("--- Full timeseries check on final data complete. ---\n")

def analyze_and_filter_missing_data():
    """
    Analyzes missing data for years >= 1951 to identify which counties to drop,
    then removes those counties from the complete dataset.
    """
    print("--- Starting County-level Filtering Logic ---")

    print(f"Using a missing year tolerance of > {MISSING_YEAR_TOLERANCE} years.") #set at the top of this file

    script_dir = get_base_path()
    input_path = os.path.join(script_dir, INPUT_FILENAME) #set at the top of this file

    if not os.path.exists(input_path):
        print(f"Error: Input file not found at: {input_path}")
        return

    print(f"Loading full dataset from: {input_path}")
    df_full = pd.read_csv(input_path)
    
    #the importer already has fips_full ('17001'), so this only fires for older files without it.
    #note: county labels printed below are therefore '17001' here, while anomalies.py writes '17_001'
    if 'fips_full' not in df_full.columns:
        df_full['fips_full'] = df_full['state'].astype(str).str.zfill(2) + '_' + df_full['county'].astype(str).str.zfill(3)

    start_year = START_YEAR #set at the top of this file
    df_analysis = df_full[df_full['year'] >= start_year].copy()
    if END_YEAR is not None:
        df_analysis = df_analysis[df_analysis['year'] <= END_YEAR] #years after END_YEAR do not count as missing (e.g. 2025 while USDA is still incomplete)

    if df_analysis.empty:
        print(f"Error: No data available from {start_year} onwards for analysis. Stopping.")
        return

    print(f"Analysis to identify problematic counties will be based on data from {start_year} to {END_YEAR if END_YEAR is not None else df_analysis['year'].max()}.")

    key_columns = [
        'value', #yield
        'GDD_GS', 'KDD_GS', 'PREC_GS',
        'CDD_1mm_GS', 'CDD_2mm_GS', 'Rx5day_GS', 'R10mm_GS', 'R20mm_GS', #precip extremes, only in the extremes importer files
        'GDD_SGF', 'KDD_SGF', 'PREC_SGF',
        'CDD_1mm_SGF', 'CDD_2mm_SGF', 'Rx5day_SGF', 'R10mm_SGF', 'R20mm_SGF'
    ]
    key_columns = [col for col in key_columns if col in df_analysis.columns]
    print(f"\nChecking for missing data in columns: {key_columns} (for 1951+ data)")

    df_analysis['is_missing_year'] = df_analysis[key_columns].isnull().any(axis=1) #same rule as above: any empty key column makes the year missing
    missing_year_counts = df_analysis.groupby('fips_full')['is_missing_year'].sum()
    
    # --- NEW: Add a diagnostic report to show the distribution of missing years ---
    print("\n--- Diagnostic Report: Missing Year Distribution (1951 onwards) ---")
    distribution = missing_year_counts.value_counts().sort_index()
    print("Num. of Missing Years | Num. of Counties")
    print("----------------------------------------")
    for years, count in distribution.items():
        if years > 0:
            print(f"{int(years):<21} | {count}")
    print("----------------------------------------\n")

    # --- CHANGED: Use the new tolerance variable for filtering ---
    counties_to_drop = missing_year_counts[missing_year_counts > MISSING_YEAR_TOLERANCE].index.tolist()
    
    print("\n--- Step 1: Identifying Counties to Drop ---")
    if counties_to_drop:
        print(f"Found {len(counties_to_drop)} counties with more than {MISSING_YEAR_TOLERANCE} years of missing data since {start_year}.")
        print("These entire counties will be removed from the full dataset.")
    else:
        print(f"No counties found with more than {MISSING_YEAR_TOLERANCE} years of missing data in the analysis period.")

    #dropping is done on the FULL dataset, so a dropped county is gone for all years, also before 1951 and after END_YEAR
    original_county_count = len(df_full['fips_full'].unique())
    df_final = df_full[~df_full['fips_full'].isin(counties_to_drop)].copy()
    retained_county_count = len(df_final['fips_full'].unique())

    print(f"\nOriginal number of counties in full dataset: {original_county_count}")
    print(f"Number of counties retained: {retained_county_count}")

    #what was done, so anomalies.py can pass it on and the run notebooks can check it
    filter_info = {
        'input_file': INPUT_FILENAME,
        'start_year': START_YEAR,
        'end_year': END_YEAR,
        'missing_year_tolerance': MISSING_YEAR_TOLERANCE,
        'key_columns': key_columns,
        'counties_in': original_county_count,
        'counties_dropped': len(counties_to_drop),
        'counties_kept': retained_county_count,
        'dropped': sorted(counties_to_drop),
    }
    return df_final, filter_info

if __name__ == '__main__':
    # This function is now mainly for a preliminary check.
    check_missing_data_by_column()
    
    # This is the main function that does the filtering.
    result = analyze_and_filter_missing_data()

    if result is not None:
        filtered_dataframe, filter_info = result
        print(f"\nFiltering complete. The returned DataFrame contains {len(filtered_dataframe)} rows.")

        # Now, run the check on the FINAL, filtered data.
        check_missing_data_full_timeseries(filtered_dataframe, start_year=START_YEAR, end_year=END_YEAR)

        # --- Save the final DataFrame to a new CSV file ---
        script_dir = get_base_path()
        output_path = os.path.join(script_dir, OUTPUT_FILENAME) #set at the top of this file

        try:
            filtered_dataframe.to_csv(output_path, index=False)
            print(f"\nSuccessfully saved the filtered data to: {output_path}")
        except Exception as e:
            print(f"\nError saving the file: {e}")

        # --- Log what was made and with which settings (panel_log.json, one entry per run) ---
        import json
        from datetime import datetime
        log_path = os.path.join(script_dir, "panel_log.json")
        log = json.load(open(log_path)) if os.path.exists(log_path) else []
        log.append({
            'created': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'step': 'missing_data',
            'output_file': OUTPUT_FILENAME,
            **filter_info,
        })
        with open(log_path, 'w') as f:
            json.dump(log, f, indent=2)
        print(f"Logged in: {log_path}")