

def get_derived_params(params):
    # set reinfection contact rate parameters
    for state in ["latent", "recovered"]:
        params["contact_rate_from_" + state] = (
                params["contact_rate"] * params["rr_infection_" + state]
        )

    params["detection_rate"] = params["case_detection_prop_sp"] / (1 - params["case_detection_prop_sp"]) * (params["natural_recovery_rate"] + params["infect_death_rate"])
    params["detection_rate_stratified"]["organ"]["e"] = params["case_detection_prop_sp"] / (10 - params["case_detection_prop_sp"]) * (params["infect_death_rate"] * params["infect_death_rate_stratified"]["organ"]["e"] + params["natural_recovery_rate"] * params["natural_recovery_rate_stratified"]["organ"]["e"]) / params["detection_rate"]
    params["detection_rate_stratified"]["organ"]["sn"] = 1 / params["detection_rate"] * (params["detection_rate"] * params["detection_rate_stratified"]["organ"]["e"] + (params["case_detection_prop"] - 0.1 * params["case_detection_prop_sp"]) * params["frontline_xpert_prop"] * (params["infect_death_rate"] * params["infect_death_rate_stratified"]["organ"]["sn"] + params["natural_recovery_rate"]))

    return params