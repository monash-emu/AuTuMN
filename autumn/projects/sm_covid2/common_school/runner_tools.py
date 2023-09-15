import yaml
import nevergrad as ng
import pymc as pm
import datetime
import arviz as az
import pandas as pd
import numpy as np
from scipy.stats import qmc
from time import sleep

import plotly.graph_objects as go
import plotly.express as px

from estival.wrappers import nevergrad as eng
from estival.wrappers import pymc as epm

import nevergrad as ng

from estival.utils.parallel import map_parallel
from estival.sampling import tools as esamp

from autumn.core.runs import ManagedRun
from autumn.infrastructure.remote import springboard

from autumn.settings.folders import PROJECTS_PATH
from autumn.projects.sm_covid2.common_school.calibration import get_bcm_object
from autumn.projects.sm_covid2.common_school.project_maker import get_school_project

from autumn.projects.sm_covid2.common_school.calibration_plots.opti_plots import plot_opti_params, plot_model_fit, plot_model_fits
from autumn.projects.sm_covid2.common_school.calibration_plots.mc_plots import make_post_mc_plots

from autumn.projects.sm_covid2.common_school.output_plots.country_spec import make_country_output_tiling

from pathlib import Path
import os
from multiprocessing import cpu_count

countries_path = Path(PROJECTS_PATH) / "sm_covid2" / "common_school" / "included_countries.yml"
with countries_path.open() as f:
    INCLUDED_COUNTRIES  = yaml.unsafe_load(f)


DEFAULT_RUN_CONFIG = {
    "n_cores": 8,
    # Opti config
    "n_opti_searches": 16,
    "opti_budget": 10000,
    # MCMC config
    "n_chains": 8,
    "metropolis_tune": 5000,
    "metropolis_draws": 30000,
    "metropolis_method": "DEMetropolisZ",
    # Full runs config
    "full_runs_samples": 1000,
    "burn_in": 20000
}

TEST_RUN_CONFIG = {
    "n_cores": 8,
    # Opti config
    "n_opti_searches": 2,
    "opti_budget": 100,
    # MCMC config
    "n_chains": 2,
    "metropolis_tune": 100,
    "metropolis_draws": 100,
    "metropolis_method": "DEMetropolisZ",
    # Full runs config
    "full_runs_samples": 50,
    "burn_in": 50
}


"""
    Functions related to model calibration
"""

def optimise_model_fit(bcm, num_workers: int = 8, warmup_iterations: int = 0, search_iterations: int = 5000, suggested_start: dict = None, opt_class=ng.optimizers.CMA):

    # Build optimizer
    full_budget = warmup_iterations + search_iterations
    opt = eng.optimize_model(bcm, obj_function=bcm.loglikelihood, suggested=suggested_start, num_workers=num_workers, opt_class=opt_class, budget=full_budget)

    # Run warm-up iterations and 
    if warmup_iterations > 0:
        res = opt.minimize(warmup_iterations)

    res = opt.minimize(search_iterations)
    best_params = res.value[1]

    # return optimal parameters and optimisation object in case we want to resume the search afterwards
    return best_params, opt


def multi_country_optimise(iso3_list: list, analysis: str = "main", num_workers: int = 8, search_iterations: int = 7000, parallel_opti_jobs: int = 4, logger=None, out_path: str = None, opt_class=ng.optimizers.CMA, best_params_dict=None):

    # perform optimisation unless best params are already provided
    if not best_params_dict:
        def country_opti_wrapper(iso3):
            bcm = get_bcm_object(iso3, analysis)
            best_p, _ = optimise_model_fit(bcm, num_workers=num_workers, warmup_iterations=0, search_iterations=search_iterations, opt_class=opt_class)                
            return best_p

        if logger:
            logger.info(f"Starting optimisation for {len(iso3_list)} countries...")
        best_params = map_parallel(country_opti_wrapper, iso3_list, n_workers=parallel_opti_jobs)
        if logger:
            logger.info("... optimisation complete.")
        
        best_params_dict = {iso3_list[i]: best_params[i] for i in range(len(iso3_list))}

        # Store optimal solutions
        if out_path:
            with open(out_path / "best_params_dict.yml", "w") as f:
                yaml.dump(best_params_dict, f)
        
        # plot optimal fits
        if not out_path:  # early return if no out_path specified
            return best_params

    if logger:
        logger.info("Start plotting optimal fits...")
    opt_fits_path = out_path / "optimised_fits"
    opt_fits_path.mkdir(exist_ok=True)
    def plot_wrapper(iso3):
        bcm = get_bcm_object(iso3, analysis)
        plot_model_fit(bcm, best_params_dict[iso3], iso3, opt_fits_path / f"best_fit_{iso3}.png")

    n_workers = cpu_count()
    map_parallel(plot_wrapper, iso3_list, n_workers=n_workers)
    if logger:
        logger.info("... finished plotting")

    return best_params_dict


def sample_with_pymc(bcm, initvals, draws=1000, tune=500, cores=8, chains=8, method="DEMetropolis", sampler_options=None):
    if method == "DEMetropolis":
        sampler = pm.DEMetropolis
    elif method == "DEMetropolisZ":
        sampler = pm.DEMetropolisZ
    else:
        raise ValueError(f"Requested sampling method '{method}' not currently supported.")

    sampler_options = sampler_options or {}

    with pm.Model() as model:    
        variables = epm.use_model(bcm)
        idata = pm.sample(step=[sampler(variables, **sampler_options)], draws=draws, tune=tune, cores=cores,chains=chains, initvals=initvals, progressbar=False)

    return idata


"""
    Functions related to post-calibration processes
"""

def extract_sample_subset(idata, n_samples, burn_in, chain_filter: list = None):
    chain_length = idata.sample_stats.sizes['draw']
    burnt_idata = idata.sel(draw=range(burn_in, chain_length))  # Discard burn-in
    
    return az.extract(burnt_idata, num_samples=n_samples)


def run_full_runs(sampled_params, iso3, analysis):

    full_runs = {}
    for scenario in ["baseline", "scenario_1"]:
        bcm = get_bcm_object(iso3, analysis, scenario)
        full_run_params = sampled_params[list(bcm.priors)]  # drop parameters for which scenario value should override calibrated value
        full_runs[scenario] = esamp.model_results_for_samples(full_run_params, bcm)

    return full_runs


def get_uncertainty_dfs(full_runs, quantiles=[.025, .25, .5, .75, .975]):
    unc_dfs = {}
    for scenario in full_runs:
        unc_df = esamp.quantiles_for_results(full_runs[scenario].results, quantiles)
        unc_df.columns.set_levels([str(q) for q in unc_df.columns.levels[1]], level=1, inplace=True)  # to avoid using floats as column names (not parquet-compatible)
        unc_dfs[scenario] = unc_df

    return unc_dfs


def calculate_diff_output_quantiles(full_runs, quantiles=[.025, .25, .5, .75, .975]):
    diff_names = {
        "cases_averted": "cumulative_incidence",
        "deaths_averted": "cumulative_infection_deaths",
        "delta_hospital_peak": "peak_hospital_occupancy",
        "delta_student_weeks_missed": "student_weeks_missed"
    }
    
    latest_time = full_runs['baseline'].results.index.max()
    
    runs_0_latest = full_runs['baseline'].results.loc[latest_time]
    runs_1_latest = full_runs['scenario_1'].results.loc[latest_time]

    abs_diff = runs_0_latest - runs_1_latest
    rel_diff = (runs_0_latest - runs_1_latest) / runs_1_latest
    
    diff_quantiles_df_abs = pd.DataFrame(
        index=quantiles, 
        data={colname: abs_diff[output_name].quantile(quantiles) for colname, output_name in diff_names.items()}
    )   
    diff_quantiles_df_rel = pd.DataFrame(
        index=quantiles, 
        data={f"{colname}_relative" : rel_diff[output_name].quantile(quantiles) for colname, output_name in diff_names.items()}
    ) 
    
    return pd.concat([diff_quantiles_df_abs, diff_quantiles_df_rel], axis=1)


"""
    Master function combining functions above
"""
def run_full_analysis(
    iso3: str, 
    analysis: str = "main",
    run_config: dict = DEFAULT_RUN_CONFIG,
    output_folder="test_outputs",
    logger=None
):
    out_path = Path(output_folder)
    assert run_config['n_chains'] <= run_config['n_opti_searches']

    # Create BayesianCompartmentalModel object
    bcm = get_bcm_object(iso3, analysis)

    """ 
        OPTIMISATION
    """
    # Sample optimisation starting points with LHS
    if logger:
        logger.info("Perform LHS sampling")
    lhs_samples = bcm.sample.lhs(run_config['n_opti_searches'])
    lhs_samples_as_dicts = lhs_samples.convert("list_of_dicts")

    # Store starting points
    with open(out_path / "LHS_init_points.yml", "w") as f:
        yaml.dump(lhs_samples_as_dicts, f)

    # Perform optimisation searches
    if logger:
        logger.info(f"Perform optimisation ({run_config['n_opti_searches']} searches)")
    n_opti_workers = 8
    def opti_func(sample_dict):
        suggested_start = {p: v for p, v in sample_dict.items() if p != 'random_process.delta_values'}
        best_p, _ = optimise_model_fit(bcm, num_workers=n_opti_workers, search_iterations=run_config['opti_budget'], suggested_start=suggested_start)
        return best_p

    best_params = map_parallel(opti_func, lhs_samples_as_dicts, n_workers=int(2 * run_config['n_cores'] / n_opti_workers))  # oversubscribing
    # Store optimal solutions
    with open(out_path / "best_params.yml", "w") as f:
        yaml.dump(best_params, f)

    if logger:
        logger.info("... optimisation completed")
    
    # Keep only n_chains best solutions and plot optimised fits
    best_outputs = esamp.model_results_for_samples(best_params, bcm, include_extras=True)
    lle, results = best_outputs.extras, best_outputs.results
    retained_indices = lle.sort_values("loglikelihood", ascending=False).index[0:run_config['n_chains']].to_list()    
    retained_best_params = [best_params[i] for i in retained_indices]
    retained_init_points = [lhs_samples_as_dicts[i] for i in retained_indices]

    # Store retained optimal solutions
    with open(out_path / "retained_best_params.yml", "w") as f:
        yaml.dump(retained_best_params, f)
    
    # Plot optimal solutions and matching starting points
    plot_opti_params(retained_init_points, retained_best_params, bcm, output_folder)

    # Plot optimised model fits on a same figure
    retained_results = results.loc[:, pd.IndexSlice[results.columns.get_level_values(1).isin(retained_indices), :]]
    plot_model_fits(retained_results, bcm, out_path / "optimal_fits.png")
    
    # Early return if MCMC not requested
    if run_config['metropolis_draws'] == 0:
        return None, None, None

    """ 
        MCMC
    """
    if logger:
        logger.info(f"Start MCMC for {run_config['metropolis_tune']} + {run_config['metropolis_draws']} iterations and {run_config['n_chains']} chains...")

    n_repeat_seed = 1
    init_vals = [[best_p] * n_repeat_seed for i, best_p in enumerate(retained_best_params)]     
    init_vals = [p_dict for sublist in init_vals for p_dict in sublist]  
    idata = sample_with_pymc(bcm, initvals=init_vals, draws=run_config['metropolis_draws'], tune=run_config['metropolis_tune'], cores=run_config['n_cores'], chains=run_config['n_chains'], method=run_config['metropolis_method'])
    idata.to_netcdf(out_path / "idata.nc")
    make_post_mc_plots(idata, run_config['burn_in'], output_folder)
    if logger:
        logger.info("... MCMC completed")
    
    """ 
        Post-MCMC processes
    """
    sampled_params = extract_sample_subset(idata, run_config['full_runs_samples'], run_config['burn_in'])  
    
    if logger:
        logger.info(f"Perform full runs for {run_config['full_runs_samples']} samples")
    full_runs = run_full_runs(sampled_params, iso3, analysis)

    if logger:
        logger.info("Calculate uncertainty quantiles") 
    unc_dfs = get_uncertainty_dfs(full_runs)
    for scenario, unc_df in unc_dfs.items():
        unc_df.to_parquet(out_path / f"uncertainty_df_{scenario}.parquet")

    if logger:
        logger.info("Calculate differential output quantiles")
    diff_quantiles_df = calculate_diff_output_quantiles(full_runs)
    diff_quantiles_df.to_parquet(out_path / "diff_quantiles_df.parquet")

    # Make multi-panel figure
    make_country_output_tiling(iso3, unc_dfs, diff_quantiles_df, output_folder)

    return idata, unc_dfs, diff_quantiles_df


"""
    Helper functions for remote runs
"""

def dump_runner_details(runner, out_folder_path):
    """
    Dumps the run_path and IP associater with a SpringboardTaskRunner object
    Args:
        runner: SpringboardTaskRunner object
        out_folder_path: pathlib Path associated with directory where data should be stored
    """

    details = {
        "run_path": runner.run_path,
        "ip": runner.instance['ip']
    }

    out_path = out_folder_path / f"{runner.run_path.split('/')[-1]}.yml"
    with out_path.open("w") as f:
        yaml.dump(details, f)


def print_continuous_status(runner, update_freq=30):
    status = runner.s3.get_status()
    print(status)

    while status in ['INIT', 'LAUNCHING']:
        sleep(update_freq)
        status = runner.s3.get_status()
    print(status)

    while status in ['RUNNING']:
        sleep(update_freq)
        status = runner.s3.get_status()
    print(status)


def download_analysis(run_path, open_out_dir=True):
    mr = ManagedRun(run_path)
    for f in mr.remote.list_contents():
        mr.remote.download(f)

    if open_out_dir:
        local_outpath = Path.home() / "Models" / "AuTuMN_new" / "data" / "outputs" / "runs" / run_path 
        os.startfile(local_outpath)