import yaml


def read_sensitivity_min_mix_res():
    res_path = "optimisation_outputs/sensitivity_min_mixing/results.yml"

    with open(res_path, "r") as yaml_file:
        results = yaml.safe_load(yaml_file)

    return results


# FIXME: this is broken
def run_sensitivity_minimum_mixing(output_dir):
    mode = "by_age"
    results = {}
    for country in Region.MIXING_OPTI_REGIONS:
        results[country] = {}
        for config in [2, 3]:
            results[country][config] = {}
            for objective in ["deaths", "yoll"]:
                results[country][config][objective] = {}
                mle_params, decision_vars = get_mle_params_and_vars(
                    output_dir, country, config, mode, objective
                )
                root_model = None
                if config == 2 and objective == "deaths":  # to run the root_model only once
                    print("running root model for " + country)
                    root_model = run_root_model(country, mle_params)
                for min_mixing in [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]:
                    modified_vars = copy.deepcopy(decision_vars)
                    modified_vars = [max([v, min_mixing]) for v in modified_vars]
                    print("evaluate objective for " + country + " " + str(config) + " " + objective)
                    h, d, yoll, p_immune, _ = objective_function(
                        modified_vars, root_model, mode, country, config, mle_params
                    )
                    res_dict = {
                        "h": bool(h),
                        "d": float(d),
                        "yoll": float(yoll),
                        "p_immune": float(p_immune),
                    }
                    results[country][config][objective][min_mixing] = res_dict

    param_file_path = "optimisation_outputs/sensitivity_min_mixing/results.yml"
    with open(param_file_path, "w") as f:
        yaml.dump(results, f)
    return results

