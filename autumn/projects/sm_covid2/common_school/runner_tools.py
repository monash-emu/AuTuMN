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

from autumn.projects.sm_covid2.common_school.calibration_plots.opti_plots import plot_opti_params, plot_model_fit, plot_multiple_model_fits
from autumn.projects.sm_covid2.common_school.calibration_plots.mc_plots import make_post_mc_plots

from autumn.projects.sm_covid2.common_school.output_plots.country_spec import make_country_output_tiling

from pathlib import Path
import os
from multiprocessing import cpu_count

countries_path = Path(PROJECTS_PATH) / "sm_covid2" / "common_school" / "included_countries.yml"
with countries_path.open() as f:
    INCLUDED_COUNTRIES  = yaml.unsafe_load(f)


DEFAULT_RUN_CONFIG = {
    "N_CORES": 8,
    "N_CHAINS": 8,
    "N_OPTI_SEARCHES": 16,
    "OPTI_BUDGET": 10000,
    "METROPOLIS_TUNE": 5000,
    "METROPOLIS_DRAWS": 30000,
    "METROPOLIS_METHOD": "DEMetropolisZ",
    "FULL_RUNS_SAMPLES": 1000,
    "BURN_IN": 20000
}


"""
    Functions related to model calibration
"""

def sample_with_lhs(n_samples, bcm):

    # sample using LHS in the right dimension
    lhs_sampled_params = [p for p in bcm.priors if p != "random_process.delta_values"]  
    d = len(lhs_sampled_params)
    sampler = qmc.LatinHypercube(d=d)
    regular_sample = sampler.random(n=n_samples)
    
    # scale the data cube to match parameter bounds
    l_bounds = [bcm.priors[p].bounds()[0] for p in lhs_sampled_params]
    u_bounds = [bcm.priors[p].bounds()[1] for p in lhs_sampled_params]
    sample = qmc.scale(regular_sample, l_bounds, u_bounds)
    
    sample_as_dicts = [{p: sample[i][j] for j, p in enumerate(lhs_sampled_params)} for i in range(n_samples)]

    return sample_as_dicts


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

def resume_opti_search(opt, extra_iterations: int = 5000):

    res = opt.minimize(extra_iterations)
    best_params = res.value[1]
    
    return best_params, opt


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


def sample_with_pymc_smc(bcm, draws=2000, cores=8, chains=4):
    
    with pm.Model() as model:    
        variables = epm.use_model(bcm)
        idata = pm.smc.sample_smc(draws=draws, chains=chains, cores=cores, progressbar=False)

    return idata


"""
    Functions related to post-calibration processes
"""

def extract_sample_subset(idata, n_samples, burn_in, chain_filter: list = None):
    chain_length = idata.sample_stats.sizes['draw']
    burnt_idata = idata.sel(draw=range(burn_in, chain_length))  # Discard burn-in
    
    return az.extract(burnt_idata, num_samples=n_samples)


def get_sampled_results(sampled_df, output_names):
    d2_index = pd.Index([index[:2] for index in sampled_df.index]).unique()

    sampled_results = {output: pd.DataFrame(index=bcm.model._get_ref_idx(), columns=d2_index) for output in output_names}

    for chain, draw in d2_index:
        # read rp delta values
        delta_values = sampled_df.loc[chain, draw]['random_process.delta_values']
        
        params_dict = sampled_df.loc[chain, draw, 0].to_dict()
        params_dict["random_process.delta_values"] = np.array(delta_values)

        run_model = bcm.run(params_dict)

        for output in output_names:
            sampled_results[output][(chain, draw)] = run_model.derived_outputs[output]

    return sampled_results


def run_full_runs(sampled_params, iso3, analysis):

    full_runs = {}
    for scenario in ["baseline", "scenario_1"]:
        bcm = get_bcm_object(iso3, analysis, scenario)
        full_run_params = sampled_params[list(bcm.priors)]  # drop parameters for which scenario value should override calibrated value
        full_runs[scenario] = esamp.model_results_for_samples(full_run_params, bcm)

    return full_runs

def diff_latest_output(outputs_df_latest_0, outputs_df_latest_1, column, relative=False):
    if not relative:     
        return outputs_df_latest_0[column] - outputs_df_latest_1[column]
    else:
        return (outputs_df_latest_0[column] - outputs_df_latest_1[column]) / outputs_df_latest_1[column]
       

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
        "death_averted": "cumulative_infection_deaths",
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
    iso3, 
    analysis="main", 
    n_cores=8,
    opti_params={'n_searches': 8, 'search_iterations': 5000},
    mcmc_params={'draws': 30000, 'tune': 5000, 'chains': 8, 'method': 'DEMetropolisZ'},
    full_run_params={'samples': 1000, 'burn_in': 20000},
    output_folder="test_outputs",
    logger=None
):
    out_path = Path(output_folder)
    assert mcmc_params['chains'] <= opti_params['n_searches']

    # Create BayesianCompartmentalModel object
    bcm = get_bcm_object(iso3, analysis)

    """ 
        OPTIMISATION
    """
    # Sample optimisation starting points with LHS
    if logger:
        logger.info("Perform LHS sampling")
    sample_as_dicts = sample_with_lhs(opti_params['n_searches'], bcm)
            
    # Store starting points
    with open(out_path / "LHS_init_points.yml", "w") as f:
        yaml.dump(sample_as_dicts, f)

    # Perform optimisation searches
    if logger:
        logger.info(f"Perform optimisation ({opti_params['n_searches']} searches)")
    n_opti_workers = 8
    def opti_func(sample_dict):
        best_p, _ = optimise_model_fit(bcm, num_workers=n_opti_workers, search_iterations=opti_params['search_iterations'], suggested_start=sample_dict)
        return best_p

    best_params = map_parallel(opti_func, sample_as_dicts, n_workers=int(2 * n_cores / n_opti_workers))  # oversubscribing
    # Store optimal solutions
    with open(out_path / "best_params.yml", "w") as f:
        yaml.dump(best_params, f)

    if logger:
        logger.info("... optimisation completed")

    # Keep only n_chains best solutions
    loglikelihoods = [bcm.loglikelihood(**p) for p in best_params]
    ll_cutoff = sorted(loglikelihoods, reverse=True)[mcmc_params['chains'] - 1]

    retained_init_points, retained_best_params = [], []
    for init_sample, best_p, ll in zip(sample_as_dicts, best_params, loglikelihoods):
        if ll >= ll_cutoff:
            retained_init_points.append(init_sample)
            retained_best_params.append(best_p)

    # Store retained optimal solutions
    with open(out_path / "retained_best_params.yml", "w") as f:
        yaml.dump(retained_best_params, f)
    
    # Plot optimal solutions and matching starting points
    plot_opti_params(retained_init_points, retained_best_params, bcm, output_folder)

    # Plot optimised model fits on a same figure
    plot_multiple_model_fits(bcm, retained_best_params, out_path / "optimal_fits.png")
    
    # Early return if MCMC not requested
    if mcmc_params['draws'] == 0:
        return None, None, None

    """ 
        MCMC
    """
    if logger:
        logger.info(f"Start MCMC for {mcmc_params['tune']} + {mcmc_params['draws']} iterations and {mcmc_params['chains']} chains...")

    n_repeat_seed = 1
    init_vals = [[best_p] * n_repeat_seed for i, best_p in enumerate(retained_best_params)]     
    init_vals = [p_dict for sublist in init_vals for p_dict in sublist]  
    idata = sample_with_pymc(bcm, initvals=init_vals, draws=mcmc_params['draws'], tune=mcmc_params['tune'], cores=n_cores, chains=mcmc_params['chains'], method=mcmc_params['method'])
    idata.to_netcdf(out_path / "idata.nc")
    make_post_mc_plots(idata, full_run_params['burn_in'], output_folder)
    if logger:
        logger.info("... MCMC completed")
    
    """ 
        Post-MCMC processes
    """
    sampled_params = extract_sample_subset(idata, full_run_params['samples'], full_run_params['burn_in'])  
    
    if logger:
        logger.info(f"Perform full runs for {full_run_params['samples']} samples")
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

    # Make multi-panel figure  #FIXME: not compatible with new unc_dfs format
    # make_country_output_tiling(iso3, unc_dfs, diff_quantiles_df, output_folder)

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