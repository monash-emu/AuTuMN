from typing import List, Dict, Optional
import copy
from summer import Stratification, Multiply

from autumn.models.sm_sir.constants import IMMUNITY_STRATA, ImmunityStratum, FlowName, Compartment
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

    # Create the immunity stratification, which applies to all compartments
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

    # The multipliers calculated from the effect of immunity only
    low_immune_effect = immunity_params.infection_risk_reduction.low
    high_immune_effect = immunity_params.infection_risk_reduction.high
    infection_adjustment = {
        ImmunityStratum.NONE: None,
        ImmunityStratum.LOW: Multiply(1. - low_immune_effect),
        ImmunityStratum.HIGH: Multiply(1. - high_immune_effect),
    }

    # Apply the adjustments to infection of the susceptibles - don't have to worry about strains here
    immunity_strat.set_flow_adjustments(
        FlowName.INFECTION,
        infection_adjustment,
    )

    # Considering people recovered from infection with each modelled strain ...
    for infected_strain in strain_strata:
        source_filter = None if infected_strain == "" else {"strain": infected_strain}

        # ... and its protection against infection with a new index strain.
        for infecting_strain in strain_strata:
            dest_filter = None if infecting_strain == "" else {"strain": infecting_strain}

            # The immunity effect for vaccine or non-cross-strain natural immunity escape properties of the strain
            non_cross_effect = 1. if infecting_strain == "" else 1. - voc_params[infecting_strain].immune_escape
            low_non_cross_multiplier = 1. - low_immune_effect * non_cross_effect
            high_non_cross_multiplier = 1. - high_immune_effect * non_cross_effect

            # The infection processes that we are adapting and for which strains may have relevance
            flows = [FlowName.EARLY_REINFECTION]
            if Compartment.WANED in compartments:
                flows.append(FlowName.LATE_REINFECTION)
            for flow in flows:

                # Cross protection from previous infection with the "infected" strain against the "infecting" strain
                cross_effect = 1. - getattr(voc_params[infected_strain].cross_protection[infecting_strain], flow) if \
                    voc_params else 1.

                # Combine the two mechanisms of protection and format for summer
                adjusters = {
                    ImmunityStratum.NONE: Multiply(cross_effect),
                    ImmunityStratum.LOW: Multiply(low_non_cross_multiplier * cross_effect),
                    ImmunityStratum.HIGH: Multiply(high_non_cross_multiplier * cross_effect),
                }

                # Apply to the stratification object
                immunity_strat.set_flow_adjustments(
                    flow,
                    adjusters,
                    source_strata=source_filter,
                    dest_strata=dest_filter,
                )

    return immunity_strat
