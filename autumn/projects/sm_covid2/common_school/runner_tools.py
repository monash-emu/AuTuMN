import os
import yaml
import nevergrad as ng
import pymc as pm
import datetime
import arviz as az
import pandas as pd
import numpy as np

import plotly.graph_objects as go
import plotly.express as px

from estival.optimization import nevergrad as eng
from estival.calibration import pymc as epm

from autumn.settings.folders import PROJECTS_PATH
from autumn.projects.sm_covid2.common_school.calibration import get_bcm_object
from autumn.projects.sm_covid2.common_school.project_maker import get_school_project

INCLUDED_COUNTRIES  = yaml.load(open(os.path.join(PROJECTS_PATH, "sm_covid2", "common_school", "included_countries.yml")), Loader=yaml.UnsafeLoader)

"""
    Functions related to model calibration
"""

def optimise_model_fit(bcm, warmup_iterations: int = 2000, search_iterations: int = 5000):

    # Build optimizer
    opt = eng.optimize_model(bcm, obj_function=bcm.loglikelihood)

    # Run warm-up iterations and 
    res = opt.minimize(warmup_iterations)

    res = opt.minimize(search_iterations)
    best_params = res.value[1]

    # return optimal parameters and optimisation object in case we want to resume the search afterwards
    return best_params, opt


def resume_opti_search(opt, extra_iterations: int = 5000):

    res = opt.minimize(extra_iterations)
    best_params = res.value[1]
    
    return best_params, opt


def sample_with_pymc(bcm, initvals, draws=1000, tune=500, cores=8, chains=8):

    with pm.Model() as model:    
        variables = epm.use_model(bcm)
        idata = pm.sample(step=[pm.DEMetropolis(variables)], draws=draws, tune=tune, cores=cores,chains=chains, initvals=initvals)

    return idata


"""
    Functions related to post-calibration processes
"""

def extract_sample_subset(idata, n_samples, burn_in):
    chain_length = idata.sample_stats.sizes['draw']
    n_chains = idata.sample_stats.sizes['chain']

    burnt_idata = idata.sel(draw=range(burn_in, chain_length))  # Discard burn-in
    calib_df = burnt_idata.to_dataframe(groups="posterior")  # Also get as dataframe

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

    output_names=["cumulative_incidence", "cumulative_infection_deaths", 'peak_hospital_occupancy', 'student_weeks_missed']

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


def diff_max_output(outputs_df, column, relative=False):
    if not relative:     
        return [
            outputs_df[(outputs_df["urun"] == urun) & (outputs_df["scenario"] == "baseline")][column].max() - 
            outputs_df[(outputs_df["urun"] == urun) & (outputs_df["scenario"] == "scenario_1")][column].max()
            for urun in outputs_df['urun'].unique()
        ]
    else:
        return [
            (outputs_df[(outputs_df["urun"] == urun) & (outputs_df["scenario"] == "baseline")][column].max() - 
            outputs_df[(outputs_df["urun"] == urun) & (outputs_df["scenario"] == "scenario_1")][column].max()) / 
            outputs_df[(outputs_df["urun"] == urun) & (outputs_df["scenario"] == "scenario_1")][column].max()
            for urun in outputs_df['urun'].unique()
        ]


def calculate_diff_outputs(outputs_df):
    index = outputs_df['urun'].unique()
    diff_outputs_df = pd.DataFrame(index=index)
    diff_outputs_df.index.name = "urun"    

    diff_outputs_df["cases_averted"] = diff_max_output(outputs_df, "cumulative_incidence")
    diff_outputs_df["cases_averted_relative"] = diff_max_output(outputs_df, "cumulative_incidence", relative=True)

    diff_outputs_df["deaths_averted"] = diff_max_output(outputs_df, "cumulative_infection_deaths")
    diff_outputs_df["deaths_averted_relative"] = diff_max_output(outputs_df, "cumulative_infection_deaths", relative=True)

    diff_outputs_df["delta_hospital_peak"] = diff_max_output(outputs_df, "peak_hospital_occupancy")
    diff_outputs_df["delta_hospital_peak_relative"] = diff_max_output(outputs_df, "peak_hospital_occupancy", relative=True)

    diff_outputs_df["delta_student_weeks_missed"] = diff_max_output(outputs_df, "student_weeks_missed")

    return diff_outputs_df


def get_quantile_outputs(outputs_df, diff_outputs_df, quantiles=[.025, .25, .5, .75, .975]):
      
    times = sorted(outputs_df.index.unique())
    scenarios = outputs_df["scenario"].unique()
    output_names = [c for c in outputs_df.columns if c not in ["scenario", "urun"]]

    uncertainty_data = []
    for scenario in scenarios:
        scenario_mask = outputs_df["scenario"] == scenario
        scenario_df = outputs_df[scenario_mask]
        for time in times:
            masked_df = scenario_df.loc[time]
            if masked_df.empty:
                continue
            for output_name in output_names:          
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

    diff_quantiles_df = diff_outputs_df.quantile(q=quantiles)

    return uncertainty_df, diff_quantiles_df


# def plot_from_model_runs_df(
#     model_results, 
#     output_name
# ) -> go.Figure:
    """
    Create interactive plot of model outputs by draw and chain
    from standard data structures.
    
    Args:
        model_results: Model outputs generated from run_samples_through_model
        sampled_df: Inference data converted to dataframe in output format of convert_idata_to_df
    """
    melted = model_results[output_name].melt(ignore_index=False)
    melted.columns = ["chain", "draw", output]
    melted.index = (melted.index  - COVID_BASE_DATETIME).days

    fig = px.line(melted, y=output, color="chain", line_group="draw", hover_data=melted.columns)

    if output_name in bcm.targets:
        fig.add_trace(
            go.Scattergl(x=bcm.targets[output_name].data.index, y=bcm.targets[output_name].data, marker=dict(color="black"), name="target", mode="markers"),
        )

    return fig