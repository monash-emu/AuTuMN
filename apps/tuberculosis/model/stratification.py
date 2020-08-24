from apps.tuberculosis.constants import Compartment, OrganStratum


def stratify_by_organ(model, params, compartments):

    compartments_to_stratify = [
        Compartment.INFECTIOUS,
    ]
    organ_strata = [
        OrganStratum.SMEAR_POSITIVE,
        OrganStratum.SMEAR_NEGATIVE,
        OrganStratum.EXTRAPULMONARY,
    ]

    # Define infectiousness adjustment by organ status
    strata_infectiousness = {}
    for stratum in organ_strata:
        if stratum + "_infect_multiplier" in params:
            strata_infectiousness[stratum] = params[stratum + "_infect_multiplier"]

    # define differential natural history by organ status
    flow_adjustments = {}
    for param_name in ['infect_death', 'recovery_rate']:
        flow_adjustments[param_name] = {}
        for organ_stratum in organ_strata:
            flow_adjustments[param_name][organ_stratum + "W"] = params[param_name + "_dict"][organ_stratum]

    # Adjust the progression rates by organ using the requested incidence proportions
    splitting_proportions = {
        "smear_positive": params['incidence_props_pulmonary'] * params["incidence_props_smear_positive_among_pulmonary"],
        "smear_negative": params['incidence_props_pulmonary'] *
                          (1. - params["incidence_props_smear_positive_among_pulmonary"]),
        "extrapulmonary": 1. - params['incidence_props_pulmonary'],
    }
    for stage in ["early", "late"]:
        flow_adjustments[stage + "_activation_rate"] = splitting_proportions

    # trigger model stratification
    model.stratify(
        "organ",
        organ_strata,
        compartments_to_stratify,
        infectiousness_adjustments=strata_infectiousness,
        flow_adjustments=flow_adjustments,
    )

