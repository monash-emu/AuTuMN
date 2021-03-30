import os
from copy import copy
from time import sleep

import yaml

from autumn.region import Region

SCENARIO_START_TIME = 440  # 15 Mar 2021

BACK_TO_NORMAL_FRACTIONS = []
MHS_REDUCTION_FRACTIONS = []
SCHOOL_REOPEN_FRACTIONS = []
VACCINE_SCENARIOS = {"mode": ["infection", "severity"], "efficacy": [0.7], "coverage": [0.13, 0.65]}


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
    sc_numbers = [float(filename.split("-")[1].split(".yml")[0]) for filename in scenario_files if filename.startswith("scenario-")]

    return int(max(sc_numbers))


def write_all_phl_scenarios(scenario_start_time=SCENARIO_START_TIME):
    clear_all_scenarios("philippines")
    sleep(1.0)

    sc_index = 0
    all_scenarios_dict = {}

    # Back to normal in workplaces and other locations
    for fraction in BACK_TO_NORMAL_FRACTIONS:
        sc_index += 1
        all_scenarios_dict[sc_index] = make_back_to_normal_sc_dict(fraction, scenario_start_time)

    # MHS reduction
    for fraction in MHS_REDUCTION_FRACTIONS:
        sc_index += 1
        all_scenarios_dict[sc_index] = make_mhs_reduction_sc_dict(fraction, scenario_start_time)

    # School reopening
    for fraction in SCHOOL_REOPEN_FRACTIONS:
        sc_index += 1
        all_scenarios_dict[sc_index] = make_school_reopen_sc_dict(fraction, scenario_start_time)

    # Vaccination
    for mode in VACCINE_SCENARIOS["mode"]:
        for coverage in VACCINE_SCENARIOS["coverage"]:
            for efficacy in VACCINE_SCENARIOS["efficacy"]:
                sc_index += 1
                all_scenarios_dict[sc_index] = make_vaccination_sc_dict(
                    mode, coverage, efficacy, scenario_start_time
                )

    # dump scenario files
    for sc_i, scenario_dict in all_scenarios_dict.items():
        print(scenario_dict["description"])

        file_path = f"params/scenario-{sc_i}.yml"
        with open(file_path, "w") as f:
            yaml.dump(scenario_dict, f)


def initialise_sc_dict(scenario_start_time):
    return {
        "parent": "apps/covid_19/regions/philippines/params/default.yml",
        "time": {"start": scenario_start_time - 2},
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


def make_vaccination_sc_dict(mode, coverage, efficacy, scenario_start_time):
    sc_dict = initialise_sc_dict(scenario_start_time)
    perc_coverage = int(100 * coverage)
    perc_efficacy = int(100 * efficacy)
    sc_dict[
        "description"
    ] = f"{perc_coverage}% coverage / {perc_efficacy}% efficacy / {mode}-preventing vaccine"

    sc_dict["vaccination"] = {
        "severity_efficacy": 0.0,
        "infection_efficacy": 0.0,
        "roll_out_components": [
            {
                "supply_period_coverage": {
                    "coverage": coverage,
                    "start_time": scenario_start_time,
                    "end_time": scenario_start_time + 270,  # 9-month roll-out
                }
            }
        ],
    }
    sc_dict["vaccination"][f"{mode}_efficacy"] = efficacy

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
            region_scenario_param["parent"] = sc_dict["parent"].replace("philippines", dir_name)

            file_path = f"../{dir_name}/params/{filename}"
            with open(file_path, "w") as f:
                yaml.dump(region_scenario_param, f)


if __name__ == "__main__":

    # Update scenarios for the Philippines app
    write_all_phl_scenarios()

    # Copy scenarios from philippines to sub-regions
    copy_scenarios_to_phl_regions()
