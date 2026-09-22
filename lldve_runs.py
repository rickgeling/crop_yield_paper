"""
Shared pieces for the lldve_run_<dataset>.ipynb notebooks.

The three run notebooks (corn, soy, corn_east100m) only differ in which master
panel they read and which runs they do. Everything they have in common lives
here, so a change is made once and not three times.

What is in here:
- read_master_panel     : read a master_panel_<dataset>.csv with the right dtypes
- check_panel_filter    : stop if the panel was filtered on another window than the notebook uses
- select_regions        : year window + re-check of the county filter (the filter itself
                          is pre_diagnostics/missing_data.py)
- impute_mY             : fill the remaining yield gaps (linear interpolation)
- report_nans           : print which regions still have NaNs in mY, mX, mZ
- run_lldve             : one run: model_builder -> LLDVE -> R2 / explained variation
- plot_coefficient_paths: quick look at the estimated beta(t) paths of one or more runs
- load_thesis_run       : read mTheta_hat of a thesis run for comparison
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from methods import LLDVE_test
from methods import lldve_evaluation
import model_builder


# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------

def read_master_panel(file_name, path=None):
    """
    Reads a master panel CSV. Codes are read as strings so leading zeros survive
    (model_builder zero-pads them again anyway, but this keeps the frame clean).
    """
    path = os.getcwd() if path is None else path
    dtype_dict = {
        'fips_full': str,
        'county': str,
        'division_yield': str,
        'division_noaa': str,
    }
    df = pd.read_csv(os.path.join(path, file_name), dtype=dtype_dict)
    print(f"Loaded {file_name}: {df.shape[0]} rows, {df['fips_full'].nunique()} counties, "
          f"years {df['year'].min()}-{df['year'].max()}")
    return df


def check_panel_filter(panel_file, start_year, end_year, max_missing_years,
                       log_path='pre_diagnostics/panel_log.json'):
    """
    The county filter lives in pre_diagnostics/missing_data.py. Every run of
    missing_data.py and anomalies.py appends an entry to pre_diagnostics/panel_log.json
    (when, which file, which settings). This looks up the latest anomalies entry
    for the panel, compares its filter settings with the notebook's, and stops if
    they differ, so a panel filtered on 1951-2024 is never modelled on 1951-2025
    or the other way round without anyone noticing.
    """
    import json
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"{log_path} not found. Run missing_data.py and anomalies.py first, they write it.")
    with open(log_path) as f:
        log = json.load(f)
    entry = None
    for e in log:   # the log is in build order, so the last match is the current panel
        if e.get('step') == 'anomalies' and e.get('output_file') == panel_file:
            entry = e
    if entry is None:
        raise ValueError(f"No anomalies entry for {panel_file} in {log_path}. Run anomalies.py for this dataset.")
    info = entry.get('filter') or {}
    if not info:
        raise ValueError(f"{panel_file} was built ({entry['created']}) without a missing_data entry, so its window is unknown. "
                         f"Run missing_data.py, then anomalies.py again.")

    problems = []
    if info['start_year'] != start_year:
        problems.append(f"start_year: panel {info['start_year']}, notebook {start_year}")
    if info['end_year'] != end_year:
        problems.append(f"end_year: panel {info['end_year']}, notebook {end_year}")
    if info['missing_year_tolerance'] != max_missing_years:
        problems.append(f"max missing years: panel {info['missing_year_tolerance']}, notebook {max_missing_years}")
    if problems:
        raise ValueError("Panel filter and notebook settings differ:\n  " + "\n  ".join(problems) +
                         "\nChange the notebook settings, or rerun missing_data.py + anomalies.py with the new window.")

    print(f"Panel filter OK: {panel_file} (built {entry['created']}) was filtered on "
          f"{info['start_year']}-{info['end_year']} with tolerance {info['missing_year_tolerance']} "
          f"({info['counties_kept']} of {info['counties_in']} counties kept).")
    return entry


def select_regions(df, start_year=1951, end_year=None, max_missing_years=4,
                   states=None, yield_col='value'):
    """
    Applies the year window and re-checks the county filter on the panel.

    The filter itself is done earlier, by pre_diagnostics/missing_data.py; the
    master panel should therefore already be clean. This function is the check:
    with the same window and tolerance nothing should be dropped here. If
    something is, the panel and the notebook are out of step (see
    check_panel_filter) and it is printed loudly.

    Rules, in this order:
    1. keep years start_year..end_year (end_year=None means everything available)
    2. optionally keep only the given states (list of 2-digit codes)
    3. drop counties that have no yield at all in the window (they would be all-NaN
       columns in mY, which the imputation cannot fix)
    4. drop counties with more than max_missing_years missing yield years in the
       window. max_missing_years=None switches this off.

    Returns the frame and a small report frame (one row per dropped county).
    """
    d = df[df['year'] >= start_year].copy()
    if end_year is not None:
        d = d[d['year'] <= end_year]
    if states is not None:
        wanted_states = []
        for s in states:
            wanted_states.append(str(s).zfill(2))
        d = d[d['fips_full'].str[:2].isin(wanted_states)]

    n_years = d['year'].nunique()
    per_county = d.groupby('fips_full')[yield_col].agg(
        n_obs='count',
    )
    per_county['n_missing'] = n_years - per_county['n_obs']

    reasons = pd.Series('', index=per_county.index)
    reasons[per_county['n_obs'] == 0] = 'no yield at all'
    if max_missing_years is not None:
        too_many = (per_county['n_obs'] > 0) & (per_county['n_missing'] > max_missing_years)
        reasons[too_many] = f'> {max_missing_years} missing years'

    dropped = per_county[reasons != ''].copy()
    dropped['reason'] = reasons[reasons != '']
    kept = per_county.index[reasons == '']

    print(f"Years {d['year'].min()}-{d['year'].max()} (T = {n_years}), "
          f"counties in: {len(per_county)}, dropped here: {len(dropped)}, kept: {len(kept)}")
    if len(dropped):
        print("!! The panel is not clean for this window. missing_data.py should have removed these:")
        print(dropped['reason'].value_counts().to_string())
    else:
        print("Nothing dropped here: the panel matches the filter (as it should).")

    return d[d['fips_full'].isin(kept)].copy(), dropped.sort_index()


def impute_mY(mY, years, method='linear'):
    """
    Fills the remaining gaps in mY (T x N) per county: linear interpolation on the
    year index, then forward/backward fill for gaps at the ends. Note that a county
    without a yield for the last year gets last year's value copied forward.
    """
    df = pd.DataFrame(mY, index=years)
    df = df.interpolate(method=method, axis=0).ffill(axis=0).bfill(axis=0)
    return df.values


def report_nans(mY, mX, mZ, regions):
    # mY: T×N
    nan_y = np.isnan(mY).any(axis=0)  # length N
    bad_y = [regions[i] for i, flag in enumerate(nan_y) if flag]
    print("mY: regions with NaNs:", bad_y or "None")

    # mX: N×T×d
    for dim in range(mX.shape[2]):
        nan_x = np.isnan(mX[:,:,dim]).any(axis=1)
        bad_x = [regions[i] for i, flag in enumerate(nan_x) if flag]
        print(f"mX (weather dim {dim}): regions with NaNs:", bad_x or "None")

    # mZ: N×T×(d+1)
    for dim in range(mZ.shape[2]):
        nan_z = np.isnan(mZ[:,:,dim]).any(axis=1)
        bad_z = [regions[i] for i, flag in enumerate(nan_z) if flag]
        print(f"mZ (regressor dim {dim}): regions with NaNs:", bad_z or "None")


# ---------------------------------------------------------------------------
# ONE RUN
# ---------------------------------------------------------------------------

def run_lldve(df, states, settings, states_label=None,
              run_r_squared=True, run_explained_variation=True):
    """
    One full run: model_builder.model_specification -> impute -> LLDVE_test.main_LLDVE
    -> optional R2 and explained-variation diagnostics.

    df            : master panel, already passed through select_regions
    states        : list of 2-digit state codes, or None for everything in df
    states_label  : label used in the output folder name; defaults to '17_18_19_27' style
    settings      : dict with the shared settings of the notebook, keys:
                    model_id, season, processing_type, saving_path,
                    bandwidth_method, manual_h, B, run_bootstrap,
                    plot_coeffs, plot_mean_fits, save_draws

    The output folder is <saving_path>/<states_label>_<model>_<season>_logY_<processing>X,
    the same tag format as the thesis results.
    """
    if states_label is None:
        states_label = "_".join(states) if states else "all_states"
    run_note = f"logY_{settings['processing_type']}X"
    full_run_label = f"{states_label}_{settings['model_id']}_{settings['season']}_{run_note}"

    print("=" * 70)
    print(f"STARTING RUN: {full_run_label}")
    print("=" * 70)

    config_options = {
        'states_to_include': states,
        'drop_region_names': [],   # county selection is done by select_regions, nothing left to drop here
        'yield_processing': {'log': True, 'detrend': False},
        'regressor_processing': settings['processing_type'],
        'model_a_squares': 'all',
        'model_1_squares': 'all',
    }

    # --- 1. build mY, mX, mZ ---
    mY, mX, mZ, years, regions, var_names, ctrl_names, var_titles = \
        model_builder.model_specification(
            df_raw_input=df.copy(),
            model_type=settings['model_id'],
            season=settings['season'],
            config_options=config_options,
        )
    mY = impute_mY(mY, years)
    report_nans(mY, mX, mZ, regions)

    # manual_h can be one number or a dict per states_label
    manual_h = settings.get('manual_h')
    if isinstance(manual_h, dict):
        manual_h = manual_h[states_label]

    # --- 2. LLDVE ---
    output_directory, results = LLDVE_test.main_LLDVE(
        mY=mY, mX=mX, mZ=mZ,
        state_identifier_str=states_label,
        model_name_str=settings['model_id'],
        season_name_str=settings['season'],
        run_note_str=run_note,
        years=years, regions=regions,
        var_names=var_names, ctrl_names=ctrl_names, var_titles=var_titles,
        output_base_dir=settings['saving_path'],
        bandwidth_selection_method=settings['bandwidth_method'],
        manual_h=manual_h,
        num_bootstrap_reps_B=settings['B'],
        PerformBootstrapAndPlotCI=settings.get('run_bootstrap', True),
        PlotSingleCoefficients=settings.get('plot_coeffs', False),
        PlotMeanFits=settings.get('plot_mean_fits', True),
        SaveBootstrapDraws=settings.get('save_draws', True),
    )
    print(f"\nLLDVE done. Results in: {output_directory}")

    # --- 3. post-estimation ---
    if results:
        if run_r_squared:
            lldve_evaluation.run_all_r_squared_analyses(
                mY_actual=mY, mY_fitted=results["fitted_values"],
                years_axis=years, regions=regions,
                output_path=output_directory, model_run_label=full_run_label,
            )
        if run_explained_variation:
            lldve_evaluation.calculate_explained_variation_decomposition(
                mTheta_hat=results["mTheta_hat"], mX_regressors=mX,
                vAlpha_hat=results["vAlpha_hat"], years_axis=years,
                output_path=output_directory, model_run_label=full_run_label,
            )

    # keep what the notebook needs for plotting / comparing
    results['years'] = years
    results['regions'] = regions
    results['var_names'] = var_names
    results['ctrl_names'] = ctrl_names
    results['label'] = full_run_label
    results['N'] = len(regions)
    results['T'] = len(years)
    print(f"--- RUN COMPLETE: {full_run_label} (N = {len(regions)}, T = {len(years)}) ---\n")
    return results


# ---------------------------------------------------------------------------
# LOOKING AT THE RESULTS
# ---------------------------------------------------------------------------

def plot_coefficient_paths(runs, title=None, thesis=None, save_path=None):
    """
    One subplot per coefficient, one line per run. `runs` is a dict
    {name: results} as returned by run_lldve. `thesis` is an optional dict
    {name: (years, mTheta_hat)} from load_thesis_run, drawn dashed for comparison.
    """
    first_run = list(runs.values())[0]
    var_names = first_run['var_names']
    n = len(var_names)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), sharex=True)
    axes = np.array(axes).reshape(-1)

    for j in range(n):
        ax = axes[j]
        for name, res in runs.items():
            ax.plot(res['years'], res['mTheta_hat'][:, j], linewidth=2, label=name)
        if thesis:
            for name, (yrs, theta) in thesis.items():
                ax.plot(yrs, theta[:, j], linewidth=1.5, linestyle='--', color='grey', label=f'{name} (thesis)')
        ax.axhline(0, color='black', linestyle=':', lw=1)
        ax.set_title(var_names[j], fontsize=11)
        ax.xaxis.set_major_locator(MultipleLocator(10))
    for k in range(n, len(axes)):
        axes[k].set_visible(False)
    axes[0].legend(fontsize=8, loc='best')
    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Saved {save_path}")
    plt.show()


def load_thesis_run(tag, thesis_dir='thesis_results/results_LLDVE_OPT', start_year=1951):
    """
    Reads mTheta_hat.csv of a thesis run and returns (years, mTheta_hat).
    The thesis panels run 1951-2024, so years are rebuilt from the row count.
    """
    theta = np.loadtxt(os.path.join(thesis_dir, tag, 'mTheta_hat.csv'), delimiter=',')
    years = np.arange(start_year, start_year + theta.shape[0])
    with open(os.path.join(thesis_dir, tag, 'h_optimal.txt')) as f:
        h = float(f.read())
    print(f"thesis {tag}: T = {theta.shape[0]}, {theta.shape[1]} coefficients, h = {h:.3f}")
    return years, theta
