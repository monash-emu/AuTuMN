from typing import List, Dict
from summer import Stratification, Multiply

from autumn.models.sm_sir.constants import IMMUNITY_STRATA, ImmunityStratum, FlowName
from autumn.models.sm_sir.parameters import Parameters


def get_immunity_strat(
        params: Parameters,
        compartments: List[str],
        vacc_immune_escape: Dict[str, float]
) -> Stratification:
    """
    This stratification is intended to capture immunity considerations other than the inter-strain immunity issues,
    which are previously considered in the strain stratification.

    Args:
        params: All model parameters
        compartments: Base model compartments
        vacc_immune_escape: The extent to which each modelled strain is a vaccine-escape

    Returns:
        Summer Stratification object to capture immunity

    """

    immunity_strat = Stratification("immunity", IMMUNITY_STRATA, compartments)

    # Set distribution of starting population
    p_immune = params.immunity_stratification.prop_immune
    p_high_among_immune = params.immunity_stratification.prop_high_among_immune
    immunity_split_props = {
        ImmunityStratum.NONE: 1. - p_immune,
        ImmunityStratum.HIGH: p_immune * p_high_among_immune,
        ImmunityStratum.LOW: p_immune * (1. - p_high_among_immune),
    }
    immunity_strat.set_population_split(immunity_split_props)

    strains = vacc_immune_escape.keys() if params.voc_emergence else ["no_strains"]

    # Consider each strain separately
    for strain in strains:

        dest_filter = None if strain == "no_strains" else {"strain": strain}

        # The modification applied to the immunity effect because of vaccine escape properties of the strain
        strain_escape = 1. - vacc_immune_escape[strain]

        # The multipliers calculated from the effect of immunity and the effect of the strain
        immunity_effects = params.immunity_stratification.infection_risk_reduction
        high_immune_multiplier = 1. - immunity_effects.high * strain_escape
        low_immune_multiplier = 1. - immunity_effects.low * strain_escape

        # Adjust infection flows based on the susceptibility of the age group
        infection_adjustment = {
            ImmunityStratum.NONE: None,
            ImmunityStratum.HIGH: Multiply(high_immune_multiplier),
            ImmunityStratum.LOW: Multiply(low_immune_multiplier)
        }

        # Apply the adjustments to all of the different infection flows implemented
        for flow_type in [FlowName.INFECTION, FlowName.EARLY_REINFECTION, FlowName.LATE_REINFECTION]:
            immunity_strat.set_flow_adjustments(flow_type, infection_adjustment, dest_strata=dest_filter)

    return immunity_strat
