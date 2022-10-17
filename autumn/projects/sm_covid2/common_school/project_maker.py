import pandas as pd
import numpy as np
from os.path import exists 
from math import ceil
from autumn.core.project import (
    Project,
    ParameterSet,
    # build_rel_path,
    get_all_available_scenario_paths,
)
from autumn.calibration import Calibration
from autumn.calibration.priors import UniformPrior
from autumn.calibration.targets import NegativeBinomialTarget, BinomialTarget
from autumn.models.sm_covid2 import get_base_params, build_model
from autumn.model_features.jax.random_process import set_up_random_process
from autumn.settings import Region, Models
from autumn.core.inputs.demography.queries import get_iso3_from_country_name
from autumn.core.inputs.database import get_input_db
from autumn.settings.constants import COVID_BASE_DATETIME

MIXING_PROXY = {"philippines": "HKG", "france": "BEL", "australia": "GBR"}

EXTRA_UNCERTAINTY_OUTPUTS = {
    "cumulative_incidence": "Cumulative number of infections",
    "transformed_random_process": "Transformed random process",
    "prop_ever_infected": "Proportion ever infected",
    "hospital_occupancy": "Hospital beds occupied with COVID-19 patients",
    "peak_hospital_occupancy": "Peak COVID-19 hospital occupancy",
    "death_missed_school_ratio": "Deaths averted per student-week of school missed",
    "abs_diff_cumulative_incidence": "COVID-19 infections averted",
    "abs_diff_cumulative_infection_deaths": "COVID-19 deaths averted",
    "rel_diff_cumulative_incidence": "COVID-19 infections averted (rel)",
    "rel_diff_cumulative_infection_deaths": "COVID-19 deaths averted (rel)",
}


diff_output_requests = [
    ["cumulative_incidence", "ABSOLUTE"],
    ["cumulative_infection_deaths", "ABSOLUTE"],
    ["cumulative_incidence", "RELATIVE"],
    ["cumulative_infection_deaths", "RELATIVE"],
]

REQUESTED_UNC_QUANTILES = [0.025, 0.25, 0.5, 0.75, 0.975]

from pathlib import Path

param_path = Path(__file__).parent.resolve() / "params"


def get_school_project(region):

    assert (
        region in Region.SCHOOL_PROJECT_REGIONS
    ), f"{region} is not registered as a school project Region"

    # Load timeseries
    timeseries = get_school_project_timeseries(region)
    # format timeseries using pandas Series
    pd_timeseries = {
        k: pd.Series(data=v["values"], index=v["times"], name=v["output_key"], dtype=float)
        for k, v in timeseries.items()
    }
    infection_deaths, cumulative_infection_deaths = (
        pd_timeseries["infection_deaths"],
        pd_timeseries["cumulative_infection_deaths"],
    )
    first_date_with_death = infection_deaths[round(infection_deaths) >= 1].index[0]

    # Get parameter set
    param_set = get_school_project_parameter_set(region, first_date_with_death)

    # Define priors
    priors = get_school_project_priors(first_date_with_death)

    # define calibration targets
    model_end_time = param_set.baseline.to_dict()["time"]["end"]
    infection_deaths_target = infection_deaths.loc[first_date_with_death:model_end_time][::14]
    cumulative_deaths_target = cumulative_infection_deaths.loc[:model_end_time][-1:]
    targets = [
        NegativeBinomialTarget(
            data=infection_deaths_target #, dispersion_param=100, #7.0
        ),  # dispersion param from Watson et al. Lancet ID
        # NegativeBinomialTarget(
        #     data=cumulative_deaths_target #, dispersion_param=100, #40.0
        # ),  # dispersion param from Watson et al. Lancet ID
        BinomialTarget(
            data=pd.Series(data=[.051], index=[199], name="prop_ever_infected") , 
            sample_sizes = [82126]
        )
    ]

    # set up random process if relevant
    if param_set.baseline.to_dict()["activate_random_process"]:
        rp_params = param_set.baseline.to_dict()["random_process"]
        rp = set_up_random_process(
            rp_params["time"]["start"],
            rp_params["time"]["end"],
            rp_params["order"],
            rp_params["time"]["step"],
        )
    else:
        rp = None

    # create calibration object
    calibration = Calibration(
        priors=priors,
        targets=targets,
        random_process=rp,
        metropolis_init="current_params",
        haario_scaling_factor=1.2, # 2.4,
        fixed_proposal_steps=500,
        metropolis_init_rel_step_size=0.02,
        using_summer2=True,
    )

    # List differential output requests
    diff_output_requests = [
        ["cumulative_incidence", "ABSOLUTE"],
        ["cumulative_infection_deaths", "ABSOLUTE"],
        ["cumulative_incidence", "RELATIVE"],
        ["cumulative_infection_deaths", "RELATIVE"],
    ]

    # create additional output to capture ratio between averted deaths and weeks of school missed
    def calc_death_missed_school_ratio(deaths_averted, student_weeks_missed):

        if student_weeks_missed[0] == 0.0:
            return np.repeat(0, deaths_averted.size)
        else:
            return deaths_averted / student_weeks_missed[0]

    post_diff_output_requests = {
        "death_missed_school_ratio": {
            "sources": ["abs_diff_cumulative_infection_deaths", "student_weeks_missed"],
            "func": calc_death_missed_school_ratio,
        }
    }

    # create the project object to be returned
    project = Project(
        region,
        Models.SM_COVID2,
        build_model,
        param_set,
        calibration,
        plots=timeseries,
        diff_output_requests=diff_output_requests,
        post_diff_output_requests=post_diff_output_requests,
    )

    return project


def get_school_project_parameter_set(region, first_date_with_death):
    """
    Get the country-specific parameter sets.

    Args:
        region: Modelled region
        first_date_with_death: first time when COVID-19 deaths were observed

    Returns:
        param_set: A ParameterSet object containing parameter sets for baseline and scenarios
    """
    # get common parameters
    base_params = get_base_params()

    common_params = base_params.update(
        param_path / "baseline.yml"
    )

    # get country-specific parameters
    country_name = region.title()
    country_params = {
        "country": {"iso3": get_iso3_from_country_name(country_name), "country_name": country_name},
        "ref_mixing_iso3": MIXING_PROXY[region],
    }

    # build full set of country-specific baseline parameters
    baseline_params = common_params.update(country_params)

    # update random process time periods according to first_date_with_death
    rp_update_params = {
        "random_process": {
            "time": {
                "start": first_date_with_death
            }
        }
    }
    baseline_params = baseline_params.update(rp_update_params)
    
    # make sure length of random process' delta_values is consistent with requested time-period
    baseline_params = resize_rp_delta_values(baseline_params)  

    # update using MLE params, if available
    mle_path= param_path / "mle_files" /  f"mle_{region}.yml"
    if exists(mle_path):
        baseline_params = baseline_params.update(mle_path, calibration_format=True)

    # get scenario parameters
    scenario_dir_path = param_path
    scenario_paths = get_all_available_scenario_paths(scenario_dir_path)
    scenario_params = [baseline_params.update(p) for p in scenario_paths]

    # build ParameterSet object
    param_set = ParameterSet(baseline=baseline_params, scenarios=scenario_params)

    return param_set


def resize_rp_delta_values(params):
    """
    Make sure that the length of the delta_values list is consistent with the requested time periods of the random process

    Args:
        params: the model parameters
    """
    rp_params = params['random_process']
    n_expected_values = ceil((rp_params['time']['end'] - rp_params['time']['start']) / rp_params['time']['step']) 
    n_passed_values = len(rp_params['delta_values'])
    if n_passed_values < n_expected_values:
        new_delta_values = rp_params['delta_values'] + [0.] * (n_expected_values - n_passed_values)
        return params.update({"random_process": {"delta_values": new_delta_values}})
    elif n_passed_values > n_expected_values:
        new_delta_values = rp_params['delta_values'][:n_expected_values]
        return params.update({"random_process": {"delta_values": new_delta_values}})
    else:
        return params
        

def get_school_project_timeseries(region):
    """
    Create a dictionarie containing country-specific timeseries. This equivalent to loading data from the timeseries json file in
    other projects.

    Args:
        region: The modelled region

    Returns:
        timeseries: A dictionary containig the timeseries
    """

    input_db = get_input_db()
    timeseries = {}
    iso3 = get_iso3_from_country_name(region.title())

    # read new daily deaths from inputs
    data = input_db.query(
        table_name="owid", conditions={"iso_code": iso3}, columns=["date", "new_deaths"]
    )

    # apply moving average
    data["smoothed_new_deaths"] = data["new_deaths"].rolling(7).mean()[6:]
    data.dropna(inplace=True)

    # add daily deaths to timeseries dict
    timeseries["infection_deaths"] = {
        "output_key": "infection_deaths",
        "title": "Daily number of deaths",
        "times": (pd.to_datetime(data["date"]) - COVID_BASE_DATETIME).dt.days.to_list(),
        "values": data["smoothed_new_deaths"].to_list(),
        "quantiles": REQUESTED_UNC_QUANTILES,
    }

    # Repeat same process for cumulated deaths
    data = input_db.query(
        table_name="owid", conditions={"iso_code": iso3}, columns=["date", "total_deaths"]
    )
    data.dropna(inplace=True)

    timeseries["cumulative_infection_deaths"] = {
        "output_key": "cumulative_infection_deaths",
        "title": "Cumulative number of deaths",
        "times": (pd.to_datetime(data["date"]) - COVID_BASE_DATETIME).dt.days.to_list(),
        "values": data["total_deaths"].to_list(),
        "quantiles": REQUESTED_UNC_QUANTILES,
    }

    # add extra derived output with no data to request uncertainty
    for output_key, title in EXTRA_UNCERTAINTY_OUTPUTS.items():
        timeseries[output_key] = {
            "output_key": output_key,
            "title": title,
            "times": [],
            "values": [],
            "quantiles": REQUESTED_UNC_QUANTILES,
        }

    return timeseries


def get_school_project_priors(first_date_with_death):
    """
    Get the list of calibration priors. This depends on the first date with death which is used to define the
    prior around the infection seed.

    Args:
        first_date_with_death: first time where deaths were reported

    Returns:
        priors: list of priors
    """

    # Work out max infectious seeding time so transmission starts before first observed deaths
    min_seed_time = 0.0
    max_seed_time = first_date_with_death - 1
    assert max_seed_time > min_seed_time, "Max seed time is lower than min seed time."

    priors = [
        UniformPrior("contact_rate", [0.01, 0.1]),
        UniformPrior("infectious_seed_time", [min_seed_time, max_seed_time]),
        UniformPrior("age_stratification.ifr.multiplier", [0.5, 1.5]),
        UniformPrior("voc_emergence.delta.new_voc_seed.time_from_gisaid_report", [-30, 30]),

        # dispersion params for targets
        UniformPrior("infection_deaths_dispersion_param", [50., 200.]),
        UniformPrior("cumulative_infection_deaths_dispersion_param", [50., 200.])        
    ]

    return priors
