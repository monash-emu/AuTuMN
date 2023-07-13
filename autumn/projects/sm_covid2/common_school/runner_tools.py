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

from autumn.core.runs import ManagedRun

from autumn.settings.folders import PROJECTS_PATH
from autumn.projects.sm_covid2.common_school.calibration import get_bcm_object
from autumn.projects.sm_covid2.common_school.project_maker import get_school_project

from autumn.projects.sm_covid2.common_school.calibration_plots.opti_plots import plot_opti_params, plot_model_fit, plot_multiple_model_fits
from autumn.projects.sm_covid2.common_school.calibration_plots.mc_plots import make_post_mc_plots

from autumn.projects.sm_covid2.common_school.output_plots.country_spec import make_country_output_tiling

from pathlib import Path
from multiprocessing import cpu_count

countries_path = Path(PROJECTS_PATH) / "sm_covid2" / "common_school" / "included_countries.yml"
with countries_path.open() as f:
    INCLUDED_COUNTRIES  = yaml.unsafe_load(f)

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


def optimise_model_fit(bcm, num_workers: int = 8, warmup_iterations: int = 2000, search_iterations: int = 5000, suggested_start: dict = None, opt_class=ng.optimizers.NGOpt):

    # Build optimizer
    opt = eng.optimize_model(bcm, obj_function=bcm.loglikelihood, suggested=suggested_start, num_workers=num_workers, opt_class=opt_class)

    # Run warm-up iterations and 
    if warmup_iterations > 0:
        res = opt.minimize(warmup_iterations)

    res = opt.minimize(search_iterations)
    best_params = res.value[1]

    # return optimal parameters and optimisation object in case we want to resume the search afterwards
    return best_params, opt


def multi_country_optimise(iso3_list: list, analysis: str = "main", num_workers: int = 8, search_iterations: int = 7000, parallel_opti_jobs: int = 4, logger=None, out_path: str = None, opt_class=ng.optimizers.NGOpt, best_params_dict=None):

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
        plot_model_fit(bcm, best_params_dict[iso3], opt_fits_path / f"best_fit_{iso3}.png")

    n_workers = cpu_count()
    map_parallel(plot_wrapper, iso3_list, n_workers=n_workers)
    if logger:
        logger.info("... finished plotting")

    return best_params_dict

def resume_opti_search(opt, extra_iterations: int = 5000):

    res = opt.minimize(extra_iterations)
    best_params = res.value[1]
    
    return best_params, opt


def sample_with_pymc(bcm, initvals, draws=1000, tune=500, cores=8, chains=8, method="DEMetropolis"):
    if method == "DEMetropolis":
        sampler = pm.DEMetropolis
    elif method == "DEMetropolisZ":
        sampler = pm.DEMetropolisZ
    else:
        raise ValueError(f"Requested sampling method '{method}' not currently supported.")

    with pm.Model() as model:    
        variables = epm.use_model(bcm)
        idata = pm.sample(step=[sampler(variables)], draws=draws, tune=tune, cores=cores,chains=chains, initvals=initvals, progressbar=False)

    return idata

"""
    Functions related to post-calibration processes
"""

def extract_sample_subset(idata, n_samples, burn_in, chain_filter: list = None):
    chain_length = idata.sample_stats.sizes['draw']
    burnt_idata = idata.sel(draw=range(burn_in, chain_length))  # Discard burn-in
    
    if chain_filter:
        burnt_idata = burnt_idata.sel(chain=chain_filter)

    param_names = list(burnt_idata.posterior.data_vars.keys())
    sampled_idata = az.extract(burnt_idata, num_samples=n_samples)  # Sample from the inference data
    sampled_df = sampled_idata.to_dataframe()[param_names]
    
    return sampled_df.sort_index(level="draw").sort_index(level="chain")


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


def run_full_runs(sampled_df, iso3, analysis):

    output_names=[
        "infection_deaths", "prop_ever_infected_age_matched", "prop_ever_infected", "cumulative_incidence",
        "cumulative_infection_deaths", "hospital_occupancy", 'peak_hospital_occupancy', 'student_weeks_missed', "transformed_random_process"
    ]

    project = get_school_project(iso3, analysis)
    default_params = project.param_set.baseline
    model = project.build_model(default_params.to_dict()) 

    scenario_params = {
        "baseline": {},
        "scenario_1": {
            "mobility": {
                "unesco_partial_opening_value": 1.,
                "unesco_full_closure_value": 1.
            }
        }
    }

    d2_index = pd.Index([index[:2] for index in sampled_df.index]).unique()

    outputs_df = pd.DataFrame(columns=output_names + ["scenario", "urun"])
    for chain, draw in d2_index:
        # read rp delta values
        update_params = sampled_df.loc[chain, draw, 0].to_dict()
        delta_values = sampled_df.loc[chain, draw]['random_process.delta_values']        
        update_params["random_process.delta_values"] = np.array(delta_values)

        params_dict = default_params.update(update_params, calibration_format=True)

        for sc_name, sc_params in scenario_params.items():
            params = params_dict.update(sc_params)          
            model.run(params.to_dict())
            derived_df = model.get_derived_outputs_df()[output_names]
            derived_df["scenario"] = [sc_name] * len(derived_df)
            derived_df["urun"] = [f"{chain}_{draw}"] * len(derived_df)
            
            outputs_df = outputs_df.append(derived_df)
    
    return outputs_df


def diff_latest_output(outputs_df_latest_0, outputs_df_latest_1, column, relative=False):
    if not relative:     
        return outputs_df_latest_0[column] - outputs_df_latest_1[column]
    else:
        return (outputs_df_latest_0[column] - outputs_df_latest_1[column]) / outputs_df_latest_1[column]
        

def calculate_diff_outputs(outputs_df):

    index = outputs_df['urun'].unique()
    latest_time = outputs_df.index.max()

    outputs_df_latest_0 = outputs_df[outputs_df['scenario'] == "baseline"].loc[latest_time]
    outputs_df_latest_0.index = outputs_df_latest_0['urun']

    outputs_df_latest_1 = outputs_df[outputs_df['scenario'] == "scenario_1"].loc[latest_time]
    outputs_df_latest_1.index = outputs_df_latest_1['urun']
    
    diff_outputs_df = pd.DataFrame(index=index)
    diff_outputs_df.index.name = "urun"    

    diff_outputs_df["cases_averted"] = diff_latest_output(outputs_df_latest_0, outputs_df_latest_1, "cumulative_incidence")
    diff_outputs_df["cases_averted_relative"] = diff_latest_output(outputs_df_latest_0, outputs_df_latest_1, "cumulative_incidence", relative=True)

    diff_outputs_df["deaths_averted"] = diff_latest_output(outputs_df_latest_0, outputs_df_latest_1, "cumulative_infection_deaths")
    diff_outputs_df["deaths_averted_relative"] = diff_latest_output(outputs_df_latest_0, outputs_df_latest_1, "cumulative_infection_deaths", relative=True)

    diff_outputs_df["delta_student_weeks_missed"] = diff_latest_output(outputs_df_latest_0, outputs_df_latest_1, "student_weeks_missed")
 
    diff_outputs_df["delta_hospital_peak"] = diff_latest_output(outputs_df_latest_0, outputs_df_latest_1, "peak_hospital_occupancy")
    diff_outputs_df["delta_hospital_peak_relative"] = diff_latest_output(outputs_df_latest_0, outputs_df_latest_1, "peak_hospital_occupancy", relative=True)

    return diff_outputs_df


def get_quantile_outputs(outputs_df, diff_outputs_df, quantiles=[.025, .25, .5, .75, .975]):
      
    times = sorted(outputs_df.index.unique())
    scenarios = outputs_df["scenario"].unique()
    unc_output_names = [
        "infection_deaths", "prop_ever_infected_age_matched", "prop_ever_infected", "transformed_random_process", "cumulative_incidence", "cumulative_infection_deaths",
        "peak_hospital_occupancy", "hospital_occupancy"
    ]

    uncertainty_data = []
    for scenario in scenarios:
        scenario_mask = outputs_df["scenario"] == scenario
        scenario_df = outputs_df[scenario_mask]
        for time in times:
            masked_df = scenario_df.loc[time]
            if masked_df.empty:
                continue
            for output_name in unc_output_names:          
                quantile_vals = np.quantile(masked_df[output_name], quantiles)
                for q_idx, q_value in enumerate(quantile_vals):
                    datum = {
                        "scenario": scenario,
                        "type": output_name,
                        "time": time,
                        "quantile": quantiles[q_idx],
                        "value": q_value,
                    }
                    uncertainty_data.append(datum)

    uncertainty_df = pd.DataFrame(uncertainty_data)

    diff_quantiles_df = pd.DataFrame(index=quantiles, data={col: np.quantile(diff_outputs_df[col], quantiles) for col in diff_outputs_df.columns})   
    
    return uncertainty_df, diff_quantiles_df


"""
    Master function combining functions above
"""
def run_full_analysis(
    iso3, 
    analysis="main", 
    opti_params={'n_searches': 8, 'num_workers': 8, 'parallel_opti_jobs': 4, 'warmup_iterations': 2000, 'search_iterations': 5000, 'init_method': "LHS"},
    mcmc_params={'draws': 10000, 'tune': 1000, 'cores': 32, 'chains': 32, 'method': 'DEMetropolis'},
    full_run_params={'samples': 1000, 'burn_in': 5000},
    output_folder="test_outputs",
    logger=None
):
    out_path = Path(output_folder)

    # Check that number of requested MCMC chains is a multiple of number of optimisation searches
    assert mcmc_params['chains'] % opti_params['n_searches'] == 0

    # Create BayesianCompartmentalModel object
    bcm = get_bcm_object(iso3, analysis)

    """ 
        OPTIMISATION
    """
    # Sample optimisation starting points with LHS
    if logger:
        logger.info("Perform LHS sampling")
    if opti_params['init_method'] == "LHS":
        sample_as_dicts = sample_with_lhs(opti_params['n_searches'], bcm)
    elif opti_params['init_method'] == "midpoint":
        sample_as_dicts = [{}] * opti_params['n_searches']
    else:
        raise ValueError('init_method optimisation argument not supported')
        
    # Store starting points
    with open(out_path / "LHS_init_points.yml", "w") as f:
        yaml.dump(sample_as_dicts, f)

    # Perform optimisation searches
    if logger:
        logger.info(f"Perform optimisation ({opti_params['n_searches']} searches)")

    def opti_func(sample_dict):
        best_p, _ = optimise_model_fit(bcm, num_workers=opti_params['num_workers'], warmup_iterations=opti_params['warmup_iterations'], search_iterations=opti_params['search_iterations'], suggested_start=sample_dict)
        return best_p

    best_params = map_parallel(opti_func, sample_as_dicts, n_workers=opti_params['parallel_opti_jobs'])

    # Store optimal solutions
    with open(out_path / "best_params.yml", "w") as f:
        yaml.dump(best_params, f)
    
    # Plot optimal solutions and starting points
    plot_opti_params(sample_as_dicts, best_params, bcm, output_folder)

    # Plot optimal model fits
    opt_fits_path = out_path / "optimised_fits"
    opt_fits_path.mkdir(exist_ok=True)
    for j, best_p in enumerate(best_params):
        plot_model_fit(bcm, best_p, opt_fits_path / f"best_fit_{j}.png")

    plot_multiple_model_fits(bcm, best_params, out_path / "optimal_fits.png")

    if logger:
        logger.info("... optimisation completed")
    
    # Early return if MCMC not requested
    if mcmc_params['draws'] == 0:
        return None, None, None

    """ 
        MCMC
    """
    if logger:
        logger.info(f"Start MCMC for {mcmc_params['tune']} + {mcmc_params['draws']} iterations and {mcmc_params['chains']} chains...")

    n_repeat_seed = int(mcmc_params['chains'] / opti_params['n_searches'])
    init_vals = [[best_p] * n_repeat_seed for i, best_p in enumerate(best_params)]     
    init_vals = [p_dict for sublist in init_vals for p_dict in sublist]  
    idata = sample_with_pymc(bcm, initvals=init_vals, draws=mcmc_params['draws'], tune=mcmc_params['tune'], cores=mcmc_params['cores'], chains=mcmc_params['chains'], method=mcmc_params['method'])
    idata.to_netcdf(out_path / "idata.nc")
    make_post_mc_plots(idata, full_run_params['burn_in'], output_folder)
    if logger:
        logger.info("... MCMC completed")
    
    """ 
        Post-MCMC processes
    """
    sample_df = extract_sample_subset(idata, full_run_params['samples'], full_run_params['burn_in'])
    if logger:
        logger.info(f"Perform full runs for {full_run_params['samples']} samples")
    outputs_df = run_full_runs(sample_df, iso3, analysis)
    if logger:
        logger.info("Calculate differential outputs")    
    diff_outputs_df = calculate_diff_outputs(outputs_df)
    if logger:
        logger.info("Calculate quantiles") 
    uncertainty_df, diff_quantiles_df = get_quantile_outputs(outputs_df, diff_outputs_df)
    uncertainty_df.to_parquet(out_path / "uncertainty_df.parquet")
    diff_quantiles_df.to_parquet(out_path / "diff_quantiles_df.parquet")

    make_country_output_tiling(iso3, uncertainty_df, diff_quantiles_df, output_folder)

    return idata, uncertainty_df, diff_quantiles_df


"""
    Helper functions for remote runs
"""

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


def download_analysis(run_path):
    mr = ManagedRun(run_path)
    for f in mr.remote.list_contents():
        mr.remote.download(f)