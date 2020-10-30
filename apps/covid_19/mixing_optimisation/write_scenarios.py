import pandas as pd
import os
import yaml

from apps import covid_19
from apps.covid_19.mixing_optimisation.mixing_opti import (
    MODES,
    DURATIONS,
    N_DECISION_VARS,
    build_params_for_phases_2_and_3,
)
from apps.covid_19.mixing_optimisation.constants import OPTI_REGIONS
from apps.covid_19.mixing_optimisation.utils import get_scenario_mapping
from autumn.constants import BASE_PATH


"""
Define scenarios for each combination of mode, duration and objective, plus the unmitigated scenario (9 scenarios)
"""
SCENARIO_MAPPING = get_scenario_mapping()
"""
Reading optimisation outputs from csv file
"""


def read_opti_outputs(output_filename):
    file_path = os.path.join(
        BASE_PATH, "apps", "covid_19", "mixing_optimisation", "optimised_variables", output_filename
    )
    df = pd.read_csv(file_path, sep=",")
    return df


def read_decision_vars(opti_outputs_df, country, mode, duration, objective):
    mask = (
        (opti_outputs_df["country"] == country)
        & (opti_outputs_df["mode"] == mode)
        & (opti_outputs_df["duration"] == duration)
        & (opti_outputs_df["objective"] == objective)
    )
    df = opti_outputs_df[mask]

    if df.empty:
        return None
    else:
        return [float(df[f"best_x{i}"]) for i in range(N_DECISION_VARS[mode])]


"""
Create dictionaries to define the optimised scenarios
"""


def build_optimised_scenario_dictionary(country, sc_idx, decision_vars, final_mixing=1.0):
    # read settings associated with scenario sc_idx
    if sc_idx == max(list(SCENARIO_MAPPING.keys())):  # this is the unmitigated scenario
        duration = DURATIONS[0]  # does not matter but needs to be defined
        mode = MODES[0]
    else:  # this is an optimised scenario
        duration = SCENARIO_MAPPING[sc_idx]["duration"]
        mode = SCENARIO_MAPPING[sc_idx]["mode"]

    # need to fetch elderly mixing reduction parameter
    app_region = covid_19.app.get_region(country)
    elderly_mixing_reduction = app_region.params['default']['elderly_mixing_reduction']

    sc_params = build_params_for_phases_2_and_3(
        decision_vars, elderly_mixing_reduction, duration, mode, final_mixing
    )
    country_folder_name = country.replace("-", "_")
    sc_params["parent"] = f"apps/covid_19/regions/{country_folder_name}/params/default.yml"
    del sc_params[
        "importation"
    ]  # Importation was used for testing during optimisation. We must remove it.

    return sc_params


def build_all_scenario_dicts_from_outputs(output_filename="dummy_vars_for_test.csv"):
    opti_outputs_df = read_opti_outputs(output_filename)

    all_sc_params = {}
    for country in OPTI_REGIONS:
        all_sc_params[country] = {}
        for sc_idx, settings in SCENARIO_MAPPING.items():
            if sc_idx == max(list(SCENARIO_MAPPING.keys())):  # this is the unmitigated scenario
                decision_vars = [1.0] * N_DECISION_VARS[MODES[0]]
            else:  # this is an optimised scenario
                decision_vars = read_decision_vars(
                    opti_outputs_df,
                    country,
                    settings["mode"],
                    settings["duration"],
                    settings["objective"],
                )
            if decision_vars is not None:
                all_sc_params[country][sc_idx] = build_optimised_scenario_dictionary(
                    country, sc_idx, decision_vars
                )
    return all_sc_params


"""
Automatically write yml files for the different scenarios and the different regions
"""


def drop_all_yml_scenario_files(all_sc_params):
    yaml.Dumper.ignore_aliases = lambda *args: True

    for country, country_sc_params in all_sc_params.items():
        for sc_idx, sc_params in country_sc_params.items():
            country_folder_name = country.replace("-", "_")
            param_file_path = f"../regions/{country_folder_name}/params/scenario-{sc_idx}.yml"
            with open(param_file_path, "w") as f:
                yaml.dump(sc_params, f)


if __name__ == "__main__":
    all_sc_params = build_all_scenario_dicts_from_outputs()
    drop_all_yml_scenario_files(all_sc_params)
