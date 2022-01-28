from typing import List, Dict, Optional
from summer import Stratification, Multiply

from autumn.models.sm_sir.constants import IMMUNITY_STRATA, ImmunityStratum, FlowName, infection_flows
from autumn.models.sm_sir.parameters import ImmunityStratification, VocComponent


def get_immunity_strat(
        compartments: List[str],
        immunity_params: ImmunityStratification,
        strain_strata: List[str],
        voc_params: Optional[Dict[str, VocComponent]],
) -> Stratification:
    """
    This stratification is intended to capture immunity considerations other than the inter-strain immunity issues,
    which are previously considered in the strain stratification. This could be thought of as vaccine-induced immunity,
    but would also capture immunity to previous strains that are not simulated in the current model.

    Args:
        compartments: Base model compartments
        immunity_params: All the immunity-related model parameters
        strain_strata: The strains being implemented in the model
        voc_params: The VoC-related parameters

    Returns:
        The summer Stratification object that captures immunity from anything other than cross-immunity between strains

    """

    immunity_strat = Stratification("immunity", IMMUNITY_STRATA, compartments)

    # Set distribution of starting population
    p_immune = immunity_params.prop_immune
    p_high_among_immune = immunity_params.prop_high_among_immune
    immunity_split_props = {
        ImmunityStratum.NONE: 1. - p_immune,
        ImmunityStratum.LOW: p_immune * (1. - p_high_among_immune),
        ImmunityStratum.HIGH: p_immune * p_high_among_immune,
    }
    immunity_strat.set_population_split(immunity_split_props)

    # Consider each strain separately
    for strain in strain_strata:

        # Allow for models in which strains are or are not being implemented
        dest_filter = None if strain == "" else {"strain": strain}

        # The modification applied to the immunity effect because of vaccine escape properties of the strain
        strain_escape = 1. if strain in ("", "wild_type") else 1. - voc_params[strain].immune_escape

        # The multipliers calculated from the effect of immunity and the effect of the strain
        immunity_effects = immunity_params.infection_risk_reduction
        high_multiplier = Multiply(1. - immunity_effects.high * strain_escape)
        low_multiplier = Multiply(1. - immunity_effects.low * strain_escape)

        # Adjust infection flows based on the susceptibility of the age group
        infection_adjustment = {
            ImmunityStratum.NONE: None,
            ImmunityStratum.HIGH: high_multiplier,
            ImmunityStratum.LOW: low_multiplier
        }

        # Apply the adjustments to all of the different infection flows implemented
        for flow_type in infection_flows:
            immunity_strat.set_flow_adjustments(flow_type, infection_adjustment, dest_strata=dest_filter)

    return immunity_strat
