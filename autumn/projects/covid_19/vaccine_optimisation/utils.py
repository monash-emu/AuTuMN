import yaml
import pandas as pd
import os

from autumn.projects.covid_19.vaccine_optimisation.vaccine_opti import initialise_opti_object
from autumn.settings import BASE_PATH


def load_decision_vars(file_name):

    file_path = os.path.join(
        BASE_PATH, "apps", "covid_19", "vaccine_optimisation", "optimal_plans", file_name
    )
    df = pd.read_csv(file_path, sep=",", header=None)
    decision_vars = list(df.iloc[0])[0:17]

    return decision_vars


def write_scenario_yml_file(country, decision_vars, sc_start_index=None):
    """
    Create a yml file for a scenario associated with a given decision vector
    """
    country_folder_name = country.replace("-", "_")

    opti_object = initialise_opti_object(country)

    # uniform scenario
    uniform_vars = [1.0 / 8.0 for _ in range(16)] + [decision_vars[-1]]

    # elderly scenario
    elderly_vars = 2 * ([0.0 for _ in range(6)] + [0.5, 0.5]) + [decision_vars[-1]]

    sc_decision_vars = [
        uniform_vars,
        elderly_vars,
        decision_vars,
    ]

    for i, _vars in enumerate(sc_decision_vars):
        sc_index = sc_start_index + i
        sc_params = opti_object.scenario_func(_vars)
        sc_params["parent"] = f"apps/covid_19/regions/{country_folder_name}/params/default.yml"

        param_file_path = os.path.join(
            BASE_PATH,
            "apps",
            "covid_19",
            "regions",
            country_folder_name,
            "params",
            f"scenario-{sc_index}.yml",
        )

        with open(param_file_path, "w") as f:
            yaml.dump(sc_params, f)


def write_optimised_scenario():
    country = "malaysia"
    file_name = "malaysia_mono_4mai2021.csv"
    sc_start_index = 1

    decision_vars = load_decision_vars(file_name)
    write_scenario_yml_file(country, decision_vars, sc_start_index)
