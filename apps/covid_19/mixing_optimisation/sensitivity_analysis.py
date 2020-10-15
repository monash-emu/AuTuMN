import os
import copy

import yaml
import pandas as pd

from .mixing_opti import run_root_model, objective_function

from apps.covid_19.mixing_optimisation.utils import prepare_table_of_param_sets
from apps.covid_19.mixing_optimisation.mixing_opti import run_sensitivity_perturbations
from apps.covid_19.mixing_optimisation.constants import OPTI_REGIONS


def main():
    common_folder_name = "Final-2020-08-04"
    burnin = {
        "france": 0,
        "belgium": 0,
        "spain": 0,
        "italy": 0,
        "sweden": 0,
        "united-kingdom": 0,
    }

    # for country in OPTI_REGIONS:
    #     path = "../../../data/outputs/calibrate/covid_19/" + country + "/" + common_folder_name
    #
    #     prepare_table_of_param_sets(path,
    #                                 country,
    #                                 n_samples=2,
    #                                 burn_in=burnin[country])

    direction = "up"  # FIXME
    target_objective = {
        "deaths": 20,
        "yoll": 1000,
    }

    for country in burnin:
        for objective in ["deaths", "yoll"]:
            for mode in ["by_age", "by_location"]:
                for config in [2, 3]:
                    print()
                    print()
                    print(
                        country + " " + objective + " " + mode + " " + str(config) + " " + direction
                    )
                    run_sensitivity_perturbations(
                        "optimisation_outputs/6Aug2020/",
                        country,
                        config,
                        mode,
                        objective,
                        target_objective_per_million=target_objective[objective],
                        tol=0.02,
                        direction=direction,
                    )


def read_csv_output_file(
    output_dir, country, config=2, mode="by_age", objective="deaths", from_streamlit=False
):
    path_to_input_csv = os.path.join("calibrated_param_sets", country + "_calibrated_params.csv")
    if from_streamlit:
        path_to_input_csv = os.path.join(
            "apps", "covid_19", "mixing_optimisation", path_to_input_csv
        )
    input_table = pd.read_csv(path_to_input_csv)

    col_names = [
        c
        for c in input_table.columns
        if c not in ["loglikelihood", "idx"] and "dispersion_param" not in c
    ]

    if mode == "by_location":
        removed_columns = ["best_x" + str(i) for i in range(3, 16)]
        col_names = [c for c in col_names if c not in removed_columns]

    output_file_name = (
        output_dir
        + "results_"
        + country
        + "_"
        + mode
        + "_"
        + str(config)
        + "_"
        + objective
        + ".csv"
    )
    out_table = pd.read_csv(output_file_name, sep=" ", header=None)
    out_table.columns = col_names
    out_table["loglikelihood"] = input_table["loglikelihood"]

    return out_table


def get_mle_params_and_vars(
    output_dir, country, config=2, mode="by_age", objective="deaths", from_streamlit=False
):

    out_table = read_csv_output_file(output_dir, country, config, mode, objective, from_streamlit)
    n_vars = {"by_age": 16, "by_location": 3}

    mle_rows = out_table[
        out_table["loglikelihood"] == out_table.loc[len(out_table) - 1, "loglikelihood"]
    ]
    mle_rows = mle_rows.sort_values(by="best_" + objective)

    decision_vars = [
        float(mle_rows.loc[mle_rows.index[0], "best_x" + str(i)]) for i in range(n_vars[mode])
    ]

    params = {}
    for c in out_table.columns:
        if c in ["idx", "loglikelihood"]:
            continue
        elif "best_" in c:
            break
        params[c] = float(mle_rows.loc[mle_rows.index[0], c])

    return params, decision_vars


def evaluate_extra_deaths(
    decision_vars,
    extra_contribution,
    i,
    root_model,
    mode,
    country,
    config,
    mle_params,
    best_objective,
    objective,
    direction="up",
):
    tested_decision_vars = copy.deepcopy(decision_vars)
    if direction == "up":
        tested_decision_vars[i] += extra_contribution
    else:
        tested_decision_vars[i] -= extra_contribution
    h, this_d, this_yoll, p_immune, m = objective_function(
        tested_decision_vars, root_model, mode, country, config, mle_params
    )
    this_objective = {"deaths": this_d, "yoll": this_yoll}

    if not h:
        delta_deaths_per_million = 1.0e6
    else:
        population = sum(m[0].compartment_values)
        delta_deaths_per_million = (this_objective[objective] - best_objective) / population * 1.0e6

    return delta_deaths_per_million


def run_sensitivity_perturbations(
    output_dir,
    country,
    config=2,
    mode="by_age",
    objective="deaths",
    target_objective_per_million=20,
    tol=0.02,
    direction="up",
):
    # target_deaths is a number of deaths per million people
    mle_params, decision_vars = get_mle_params_and_vars(
        output_dir, country, config, mode, objective
    )
    root_model = run_root_model(country, mle_params)

    h, best_d, best_yoll, p_immune, m = objective_function(
        decision_vars, root_model, mode, country, config, mle_params
    )
    best_objective = {
        "deaths": best_d,
        "yoll": best_yoll,
    }

    delta_contributions = []
    for i in range(len(decision_vars)):
        print("Age group " + str(i))
        extra_contribution_lower = 0.0
        if direction == "up":
            extra_contribution_upper = 1.0 - decision_vars[i]
        else:
            extra_contribution_upper = decision_vars[i]

        if extra_contribution_upper < tol:
            best_solution = extra_contribution_upper if direction == "up" else decision_vars[i]
        else:
            # find an upper bound (lower if direction is down):
            delta_deaths_per_million = evaluate_extra_deaths(
                decision_vars,
                extra_contribution_upper,
                i,
                root_model,
                mode,
                country,
                config,
                mle_params,
                best_objective[objective],
                objective,
                direction,
            )
            if delta_deaths_per_million < target_objective_per_million:
                best_solution = extra_contribution_upper
            else:
                loop_count = 0
                while (extra_contribution_upper - extra_contribution_lower) > tol:
                    evaluation_point = (extra_contribution_lower + extra_contribution_upper) / 2.0
                    delta_deaths_per_million = evaluate_extra_deaths(
                        decision_vars,
                        evaluation_point,
                        i,
                        root_model,
                        mode,
                        country,
                        config,
                        mle_params,
                        best_objective[objective],
                        objective,
                        direction,
                    )
                    if delta_deaths_per_million > target_objective_per_million:
                        extra_contribution_upper = evaluation_point
                    else:
                        extra_contribution_lower = evaluation_point
                    loop_count += 1
                    if loop_count >= 20:
                        print("FLAG INFINITE LOOP")
                        break

                if (extra_contribution_upper - target_objective_per_million) < (
                    target_objective_per_million - extra_contribution_lower
                ):
                    best_solution = extra_contribution_upper
                else:
                    best_solution = extra_contribution_lower

        delta_contributions.append(best_solution)

        print(best_solution)
    output_file_path = os.path.join(
        "optimisation_outputs",
        "sensitivity",
        country + "_" + mode + "_" + str(config) + "_" + objective + "_" + direction + ".yml",
    )

    with open(output_file_path, "w") as f:
        yaml.dump(delta_contributions, f)


if __name__ == "__main__":
    main()