"""
The scenario parameter files of the "philippines" application 
can be replicated for the regional philippines applications.

To do this, run this script.
"""
import os
from copy import copy
from time import sleep

import yaml

from autumn.settings import Region

SCENARIO_START_TIME = 559  # 12 Jul 2021
#
# WORKFORCE_PROP = []
# BACK_TO_NORMAL_FRACTIONS = []
# MHS_REDUCTION_FRACTIONS = []
# SCHOOL_REOPEN_FRACTIONS = []

BASELINE_TARGET_VACC_COVERAGE = .3
VACCINE_SCENARIOS = {"extra_coverage_from_baseline_target": [0., .4]}
INCREASED_MOBILITY = [0., .3, .5]
INCREASED_TESTING = [0., .5]


def clear_all_scenarios(region):
    dir_name = region.replace("-", "_")
    file_path = f"../{dir_name}/params/"

    scenario_files = os.listdir(file_path)
    for filename in scenario_files:
        if filename.startswith("scenario-"):
            os.remove(f"../{dir_name}/params/{filename}")


def get_greater_scenario_number(region):
    dir_name = region.replace("-", "_")
    file_path = f"../{dir_name}/params/"

    scenario_files = os.listdir(file_path)
    sc_numbers = [
        float(filename.split("-")[1].split(".yml")[0])
        for filename in scenario_files
        if filename.startswith("scenario-")
    ]

    return int(max(sc_numbers))


def write_all_phl_scenarios(scenario_start_time=SCENARIO_START_TIME):
    clear_all_scenarios("philippines")
    sleep(1.0)

    sc_index = 0
    all_scenarios_dict = {}

    # # Back to normal in workplaces and other locations
    # for fraction in BACK_TO_NORMAL_FRACTIONS:
    #     sc_index += 1
    #     all_scenarios_dict[sc_index] = make_back_to_normal_sc_dict(fraction, scenario_start_time)
    #
    # # MHS reduction
    # for fraction in MHS_REDUCTION_FRACTIONS:
    #     sc_index += 1
    #     all_scenarios_dict[sc_index] = make_mhs_reduction_sc_dict(fraction, scenario_start_time)
    #
    # # School reopening
    # for fraction in SCHOOL_REOPEN_FRACTIONS:
    #     sc_index += 1
    #     all_scenarios_dict[sc_index] = make_school_reopen_sc_dict(fraction, scenario_start_time)

    # Vaccination combined with mobility changes
    for extra_coverage in VACCINE_SCENARIOS["extra_coverage_from_baseline_target"]:
        for increased_mobility in INCREASED_MOBILITY:
            for increased_testing in INCREASED_TESTING:
                if extra_coverage == 0. and increased_mobility == 0. and increased_testing == 0.:
                    continue  # this is the baseline scenario
                sc_index += 1
                all_scenarios_dict[sc_index] = make_vaccination_and_increased_mobility_and_increased_testing_sc_dict(
                    extra_coverage, increased_mobility, increased_testing, scenario_start_time
                )

    # dump scenario files
    for sc_i, scenario_dict in all_scenarios_dict.items():
        print(scenario_dict["description"])

        file_path = f"params/scenario-{sc_i}.yml"
        with open(file_path, "w") as f:
            yaml.dump(scenario_dict, f)


def initialise_sc_dict(scenario_start_time):
    return {
        "time": {"start": scenario_start_time},
    }


def make_back_to_normal_sc_dict(fraction, scenario_start_time):
    sc_dict = initialise_sc_dict(scenario_start_time)
    perc = int(100 * fraction)
    sc_dict["description"] = f"{perc}% return to normal (work and other locations)"

    sc_dict["mobility"] = {
        "mixing": {
            "work": {
                "append": True,
                "times": [scenario_start_time],
                "values": [["close_gap_to_1", fraction]],
            },
            "other_locations": {
                "append": True,
                "times": [scenario_start_time],
                "values": [["close_gap_to_1", fraction]],
            },
        }
    }

    return sc_dict


def make_mhs_reduction_sc_dict(fraction, scenario_start_time):
    sc_dict = initialise_sc_dict(scenario_start_time)
    perc = int(100 * fraction)
    sc_dict["description"] = f"Reduction in MHS by {perc}%"

    sc_dict["mobility"] = {
        "microdistancing": {
            "behaviour": {
                "parameters": {
                    "times": [scenario_start_time - 1, scenario_start_time],
                    "values": [1.0, 1.0 - fraction],
                }
            }
        }
    }

    return sc_dict


def make_school_reopen_sc_dict(fraction, scenario_start_time):
    sc_dict = initialise_sc_dict(scenario_start_time)
    perc = int(100 * fraction)
    sc_dict["description"] = f"{perc}% of schools reopen"

    sc_dict["mobility"] = {
        "mixing": {
            "school": {
                "append": False,
                "times": [scenario_start_time - 1, scenario_start_time],
                "values": [0.0, fraction],
            },
        }
    }

    return sc_dict


def make_vaccination_and_workforce_sc_dict(coverage, prop_workforce, scenario_start_time):
    sc_dict = initialise_sc_dict(scenario_start_time)
    perc_coverage = int(100 * coverage)
    perc_workforce = int(100 * prop_workforce)

    sc_dict[
        "description"
    ] = f"{perc_coverage}% vaccine coverage / {perc_workforce}% onsite workers"

    sc_dict["vaccination"] = {
        "roll_out_components": [
            {
                "supply_period_coverage": {
                    "coverage": coverage,
                    "start_time": scenario_start_time,
                    "end_time": 731,  # end of year 2021
                }
            }
        ],
    }

    sc_dict["mobility"] = {
        "mixing": {
            "work": {
                "append": True,
                "times": [scenario_start_time - 1, scenario_start_time + 1],
                "values": [["repeat_prev"], prop_workforce],
            },
        }
    }

    return sc_dict


def make_vaccination_and_increased_mobility_and_increased_testing_sc_dict(
        extra_coverage, increased_mobility, increased_testing, scenario_start_time
):
    sc_dict = initialise_sc_dict(scenario_start_time)
    perc_coverage = int(100 * (extra_coverage + BASELINE_TARGET_VACC_COVERAGE))
    perc_increase_mobility = int(100 * increased_mobility)
    perc_increase_testing = int(100 * increased_testing)

    mobility_description = f"{perc_increase_mobility}% increased mobility" if perc_increase_mobility > 0. else "baseline mobility"
    testing_description = f"{perc_increase_testing}% increased testing" if perc_increase_testing > 0. else "baseline testing"

    sc_dict[
        "description"
    ] = f"{perc_coverage}% vaccine coverage / {mobility_description} / {testing_description}"

    if extra_coverage > 0.:
        sc_dict["vaccination"] = {
            "roll_out_components": [
                {
                    "supply_period_coverage": {
                        "coverage": extra_coverage + BASELINE_TARGET_VACC_COVERAGE,
                        "start_time": scenario_start_time,
                        "end_time": 731,  # end of year 2021
                    }
                }
            ],
        }
    if increased_mobility > 0.:
        sc_dict["mobility"] = {
            "mixing": {
                "work": {
                    "append": True,
                    "times": [scenario_start_time - 1, scenario_start_time + 1],
                    "values": [["repeat_prev"], ["scale_prev", 1. + increased_mobility]],
                },
                "other_locations": {
                    "append": True,
                    "times": [scenario_start_time - 1, scenario_start_time + 1],
                    "values": [["repeat_prev"], ["scale_prev", 1. + increased_mobility]],
                },
            }
        }

    if increased_testing > 0.:
        sc_dict['testing_to_detection'] = {
            'test_multiplier': {
                'times': [scenario_start_time - 1, scenario_start_time + 1],
                'values': [1., 1. + increased_testing]
            }
        }

    return sc_dict


def read_all_phl_scenarios():
    """
    Read all the scenarios defined for the "philippines" application
    :return: a dictionary containing all the scenario parameters
    """
    scenario_param_dicts = {}

    param_files = os.listdir("params/")
    for filename in param_files:
        if filename.startswith("scenario-"):
            file_path = f"params/{filename}"
            with open(file_path) as file:
                sc_dict = yaml.load(file)

            scenario_param_dicts[filename] = sc_dict

    return scenario_param_dicts


def copy_scenarios_to_phl_regions():
    """
    Replicate all scenarios defined for the "philippines" application to the three regional applications
    :return:
    """
    scenario_param_dicts = read_all_phl_scenarios()

    for region in Region.PHILIPPINES_REGIONS:
        if region == "philippines":
            continue
        dir_name = region.replace("-", "_")

        clear_all_scenarios(region)
        sleep(1.0)

        for filename, sc_dict in scenario_param_dicts.items():
            region_scenario_param = copy(sc_dict)
            file_path = f"../{dir_name}/params/{filename}"
            with open(file_path, "w") as f:
                yaml.dump(region_scenario_param, f)


if __name__ == "__main__":

    # Update scenarios for the Philippines app
    write_all_phl_scenarios()

    # Copy scenarios from philippines to sub-regions
    copy_scenarios_to_phl_regions()
