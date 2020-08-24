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

    flow_adjustments = {}  # FIXME

    model.stratify(
        "organ",
        organ_strata,
        compartments_to_stratify,
        infectiousness_adjustments=strata_infectiousness,
        flow_adjustments=flow_adjustments,
    )

