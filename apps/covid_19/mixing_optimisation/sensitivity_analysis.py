import os
import copy

import yaml

from apps.covid_19.mixing_optimisation.constants import OPTI_REGIONS
from apps.covid_19.mixing_optimisation.mixing_opti import MODES, DURATIONS, OBJECTIVES, run_root_model, objective_function

from apps.covid_19.mixing_optimisation.write_scenarios import read_opti_outputs, read_decision_vars
from autumn.constants import BASE_PATH


def main():
    opti_output_filename = "dummy_vars_for_test.csv"
    opti_outputs_df = read_opti_outputs(opti_output_filename)
    target_objective = {
        "deaths": 20,
        "yoll": 1000,
    }

    for direction in ["up", "down"]:
        for country in OPTI_REGIONS:
            for mode in MODES:
                for duration in DURATIONS:
                    for objective in OBJECTIVES:
                        print()
                        print()
                        print(
                            country + " " + objective + " " + mode + " " + str(duration) + " " + direction
                        )
                        run_sensitivity_perturbations(
                            opti_outputs_df,
                            country,
                            duration,
                            mode,
                            objective,
                            target_objective_per_million=target_objective[objective],
                            tol=0.02,
                            direction=direction,
                        )


def evaluate_extra_deaths(
    decision_vars,
    extra_contribution,
    i,
    root_model,
    mode,
    country,
    duration,
    best_objective,
    objective,
    direction="up",
):
    tested_decision_vars = copy.deepcopy(decision_vars)
    if direction == "up":
        tested_decision_vars[i] += extra_contribution
    else:
        tested_decision_vars[i] -= extra_contribution
    h, this_d, this_yoll = objective_function(
        tested_decision_vars, root_model, mode, country, duration, called_from_sensitivity_analysis=True
    )
    this_objective = {"deaths": this_d, "yoll": this_yoll}

    if not h:
        delta_deaths_per_million = 1.0e6
    else:
        population = sum(m[0].compartment_values)  # FIXME 
        delta_deaths_per_million = (this_objective[objective] - best_objective) / population * 1.0e6

    return delta_deaths_per_million


def run_sensitivity_perturbations(
    opti_outputs_df,
    country,
    duration="six_months",
    mode="by_age",
    objective="deaths",
    target_objective_per_million=20,
    tol=0.02,
    direction="up",
):
    # target_deaths is a number of deaths per million people
    decision_vars = read_decision_vars(opti_outputs_df, country, mode, duration, objective)

    if decision_vars is None:
        return

    root_model = run_root_model(country)

    h, best_d, best_yoll = objective_function(
        decision_vars, root_model, mode, country, duration, called_from_sensitivity_analysis=True
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
                duration,
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
                        duration,
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
        BASE_PATH, "apps", "covid_19", "mixing_optimisation", "optimised_variables",
        "optimal_plan_sensitivity",
        country + "_" + mode + "_" + duration + "_" + objective + "_" + direction + ".yml",
    )

    with open(output_file_path, "w") as f:
        yaml.dump(delta_contributions, f)


if __name__ == "__main__":
    main()
