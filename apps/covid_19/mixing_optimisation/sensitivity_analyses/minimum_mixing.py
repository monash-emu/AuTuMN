import os
import copy
import yaml

from apps.covid_19.mixing_optimisation.constants import OPTI_REGIONS
from apps.covid_19.mixing_optimisation.mixing_opti import MODES, DURATIONS, OBJECTIVES, run_root_model, objective_function
from apps.covid_19.mixing_optimisation.write_scenarios import read_opti_outputs, read_decision_vars
from autumn.constants import BASE_PATH


def main():
    opti_output_filename = "dummy_vars_for_test.csv"
    run_sensitivity_minimum_mixing(opti_output_filename)


def read_sensitivity_min_mix_res():
    file_path = os.path.join(
        BASE_PATH, "apps", "covid_19", "mixing_optimisation", "sensitivity_analyses", "min_mixing_results.yml"
    )

    with open(file_path, "r") as yaml_file:
        results = yaml.safe_load(yaml_file)

    return results


def run_sensitivity_minimum_mixing(opti_output_filename="dummy_vars_for_test.csv"):
    opti_outputs_df = read_opti_outputs(opti_output_filename)
    results = {}
    for country in OPTI_REGIONS:
        results[country] = {}
        root_model = run_root_model(country)
        for mode in MODES:
            results[country][mode] = {}
            for duration in DURATIONS:
                results[country][mode][duration] = {}
                for objective in OBJECTIVES:
                    results[country][mode][duration][objective] = {}
                    decision_vars = read_decision_vars(opti_outputs_df, country, mode, duration, objective)
                    if decision_vars is None:
                        continue
                    for min_mixing in [0.10, 0.20, 0.30, 0.40, 0.50]:
                        modified_vars = copy.deepcopy(decision_vars)
                        modified_vars = [max([v, min_mixing]) for v in modified_vars]
                        print(f"evaluate objective for {country} | {mode} | {duration} | {objective}: min_mixing={min_mixing}")
                        h, d, yoll = objective_function(
                            modified_vars, root_model, mode, country, duration
                        )
                        res_dict = {
                            "h": bool(h),
                            "d": float(d),
                            "yoll": float(yoll),
                        }
                        results[country][mode][duration][objective][min_mixing] = res_dict
    file_path = os.path.join(
        BASE_PATH, "apps", "covid_19", "mixing_optimisation", "sensitivity_analyses", "min_mixing_results.yml"
    )
    with open(file_path, "w") as f:
        yaml.dump(results, f)
    return results


if __name__ == "__main__":
    main()

