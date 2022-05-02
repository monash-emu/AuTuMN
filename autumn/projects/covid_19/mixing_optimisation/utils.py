import os

import numpy as np

from autumn.projects.covid_19.mixing_optimisation.mixing_opti import DURATIONS, MODES, OBJECTIVES
from autumn.tools import inputs

HOSPITAL_DATA_DIR = os.path.join("hospitalisation_data")
country_mapping = {"united-kingdom": "The United Kingdom"}


def get_prior_distributions_for_opti():
    prior_list = [
        {
            "param_name": "contact_rate",
            "distribution": "uniform",
            "distri_params": [0.03, 0.07],
        },
        {
            "param_name": "infectious_seed",
            "distribution": "uniform",
            "distri_params": [50., 600.],
        },
        {
            "param_name": "sojourn.compartment_periods_calculated.exposed.total_period",
            "distribution": "trunc_normal",
            "distri_params": [5.5, 0.97],
            "trunc_range": [1.0, np.inf],
        },
        {
            "param_name": "sojourn.compartment_periods_calculated.active.total_period",
            "distribution": "trunc_normal",
            "distri_params": [6.5, 0.77],
            "trunc_range": [1.0, np.inf],
        },
        {
            "param_name": "infection_fatality.multiplier",
            "distribution": "uniform",
            "distri_params": [0.5, 3.8],  # 3.8 to match the highest value found in Levin et al.
        },
        {
            "param_name": "testing_to_detection.assumed_cdr_parameter",
            "distribution": "uniform",
            "distri_params": [0.02, 0.20],
        },
        # vary symptomatic and hospitalised proportions
        {
            "param_name": "clinical_stratification.props.symptomatic.multiplier",
            "distribution": "uniform",
            "distri_params": [0.6, 1.4],
        },
        {
            "param_name": "clinical_stratification.props.hospital.multiplier",
            "distribution": "uniform",
            "distri_params": [0.5, 1.5],
        },
        # Micro-distancing
        {
            "param_name": "mobility.microdistancing.behaviour.parameters.inflection_time",
            "distribution": "uniform",
            "distri_params": [60, 130],
        },
        {
            "param_name": "mobility.microdistancing.behaviour.parameters.end_asymptote",
            "distribution": "uniform",
            "distri_params": [0.25, 0.80],
        },
        # {
        #     "param_name": "mobility.microdistancing.behaviour_adjuster.parameters.inflection_time",
        #     "distribution": "uniform",
        #     "distri_params": [130, 250],
        # },
        # {
        #     "param_name": "mobility.microdistancing.behaviour_adjuster.parameters.start_asymptote",
        #     "distribution": "uniform",
        #     "distri_params": [0.4, 1.0],
        # },
        # {
        #     "param_name": "elderly_mixing_reduction.relative_reduction",
        #     "distribution": "uniform",
        #     "distri_params": [0.0, 0.5],
        # },
    ]
    return prior_list


def get_weekly_summed_targets(times, values):
    assert len(times) == len(values), "times and values must have the same length"
    assert len(times) >= 7, "number of time points must be greater than 7 to compute weekly data"

    t_low = min(times)
    t_max = max(times)

    w_times = []
    w_values = []
    while t_low < t_max:
        this_week_indices = [i for i, t in enumerate(times) if t_low <= t < t_low + 7]
        this_week_times = [times[i] for i in this_week_indices]
        this_week_values = [values[i] for i in this_week_indices]
        w_times.append(round(np.mean(this_week_times)))
        w_values.append(np.mean(this_week_values))
        t_low += 7

    return w_times, w_values


def get_country_population_size(country):
    iso_3 = inputs.demography.queries.get_iso3_from_country_name(country)
    return sum(inputs.get_population_by_agegroup(["0"], iso_3, None, year=2020))


def get_scenario_mapping():
    scenario_mapping = {}
    _sc_idx = 1
    for _mode in MODES:
        for _duration in DURATIONS:
            for _objective in OBJECTIVES:
                scenario_mapping[_sc_idx] = {
                    "mode": _mode,
                    "duration": _duration,
                    "objective": _objective,
                }
                _sc_idx += 1
    scenario_mapping[_sc_idx] = {
        "mode": None,
        "duration": None,
        "objective": None,
    }  # extra scenario for unmitigated

    return scenario_mapping


def get_scenario_mapping_reverse(mode, duration, objective):

    scenario_mapping = get_scenario_mapping()
    found_sc_idx = False
    for _sc_idx, settings in scenario_mapping.items():
        if (
            settings["mode"] == mode
            and settings["duration"] == duration
            and settings["objective"] == objective
        ):
            found_sc_idx = True
            return _sc_idx

    if not found_sc_idx:
        return None


def get_wi_scenario_mapping(vary_final_mixing=False):
    final_mixings = [1.0, 0.9, 0.8, 0.7] if vary_final_mixing else [1.0]
    scenario_mapping = {}
    _sc_idx = 1
    for _final_mixing in final_mixings:
        for _duration in DURATIONS:
            for _objective in OBJECTIVES:
                scenario_mapping[_sc_idx] = {
                    "duration": _duration,
                    "objective": _objective,
                    "final_mixing": _final_mixing,
                }
                _sc_idx += 1

    return scenario_mapping


def get_wi_scenario_mapping_reverse(duration, objective, final_mixing=1.0):

    wi_scenario_mapping = get_wi_scenario_mapping(vary_final_mixing=True)

    found_wi_sc_idx = False
    for _wi_sc_idx, settings in wi_scenario_mapping.items():
        if (
            settings["final_mixing"] == final_mixing
            and settings["duration"] == duration
            and settings["objective"] == objective
        ):
            found_wi_sc_idx = True
            return _wi_sc_idx

    if not found_wi_sc_idx:
        return None
