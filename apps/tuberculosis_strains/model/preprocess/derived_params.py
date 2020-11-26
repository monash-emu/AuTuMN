

def get_derived_params(params):
    # set reinfection contact rate parameters
    for state in ["latent", "recovered"]:
        params["contact_rate_from_" + state] = (
                params["contact_rate"] * params["rr_infection_" + state]
        )

    params["detection_rate"] = params["case_detection_prop_sp"] / (1 - params["case_detection_prop_sp"]) * (params["self_recovery_rate"] * params["self_recovery_rate_stratified"]["organ"]["sp"] + params["infect_death_rate"] * params["infect_death_rate_stratified"]["organ"]["sp"])
    params["detection_rate_stratified"]["organ"]["sp"] = 1.0
    params["detection_rate_stratified"]["organ"]["e"] = params["case_detection_prop_sp"] / (2 - params["case_detection_prop_sp"]) * (params["infect_death_rate"] * params["infect_death_rate_stratified"]["organ"]["e"] + params["self_recovery_rate"] * params["self_recovery_rate_stratified"]["organ"]["e"]) / params["detection_rate"]
    params["detection_rate_stratified"]["organ"]["sn"] = 1 / params["detection_rate"] * (params["detection_rate"] * params["detection_rate_stratified"]["organ"]["e"] + (params["case_detection_prop"] - 0.5 * params["case_detection_prop_sp"]) * params["frontline_xpert_prop"] * (params["infect_death_rate"] * params["infect_death_rate_stratified"]["organ"]["sn"] + params["self_recovery_rate"] * params["self_recovery_rate_stratified"]["organ"]["sp"]))

    params["treatment_recovery_rate"] = params["treatment_success_prop"] / params["treatment_duration"]
    params["treatment_recovery_rate_stratified"]["strain"]["ds"] = 1.0
    params["treatment_recovery_rate_stratified"]["strain"]["mdr"] = (params["treatment_success_prop"] * params["treatment_success_prop_stratified"]["strain"]["mdr"] / (params["treatment_duration"] * params["treatment_duration_stratified"]["strain"]["mdr"])) / params["treatment_recovery_rate"]

    params["treatment_death_rate"] = params["treatment_mortality_prop"] / params["treatment_duration"]
    params["treatment_death_rate_stratified"]["strain"]["ds"] = 1.0
    params["treatment_death_rate_stratified"]["strain"]["mdr"] = ( params["treatment_mortality_prop"] * params["treatment_mortality_prop_stratified"]["strain"]["mdr"] / (params["treatment_duration"] * params["treatment_duration_stratified"]["strain"]["mdr"] ) ) / params["treatment_death_rate"]

    params["treatment_default_rate"] = params["treatment_default_prop"] / params["treatment_duration"]
    params["treatment_default_rate_stratified"]["strain"]["ds"] = 1 - params["amplification_prob"]
    params["treatment_default_rate_stratified"]["strain"]["mdr"] = ( params["treatment_default_prop"] * params["treatment_default_prop_stratified"]["strain"]["mdr"] / (params["treatment_duration"] * params["treatment_duration_stratified"]["strain"]["mdr"] ) ) / params["treatment_default_rate"]

    params["treatment_default_rate_stratified"]["classified"]["correctly"] = 1.0
    params["treatment_default_rate_stratified"]["classified"]["incorrectly"] = params["proportion_mdr_misdiagnosed_as_ds_transition_to_fail_lost"] / (params["treatment_default_rate"] * params["treatment_default_rate_stratified"]["strain"]["mdr"])

    params["amplification_rate"] = params["treatment_default_rate"] * params["amplification_prob"]

    params["reinfection_rate"] = params["contact_rate"] * params["rr_infection_latent"]

    return params