from typing import List, Dict, Optional
from summer import Stratification, Multiply

from autumn.models.sm_sir.constants import IMMUNITY_STRATA, ImmunityStratum, FlowName
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

    # The multipliers calculated from the effect of immunity only
    immunity_effects = immunity_params.infection_risk_reduction
    high_multiplier = 1. - immunity_effects.high
    low_multiplier = 1. - immunity_effects.low

    # Adjust infection flows based on the susceptibility of the age group
    infection_adjustment = {
        ImmunityStratum.NONE: 1.,
        ImmunityStratum.HIGH: high_multiplier,
        ImmunityStratum.LOW: low_multiplier
    }

    # Apply the adjustments to all of the different infection flows implemented
    immunity_strat.set_flow_adjustments(
        FlowName.INFECTION,
        {strat: Multiply(multiplier) for strat, multiplier in infection_adjustment.items()}
    )

    # Considering recovery with one particular modelled strain ...
    for infected_strain, infected_strain_params in voc_params.items():

        # The modification applied to the immunity effect because of vaccine escape properties of the strain
        strain_escape = 1. if infected_strain in ("", "wild_type") else 1. - voc_params[infected_strain].immune_escape

        # The multipliers calculated from the effect of immunity and the effect of the strain
        immunity_effects = immunity_params.infection_risk_reduction
        low_multiplier = 1. - immunity_effects.low * strain_escape
        high_multiplier = 1. - immunity_effects.high * strain_escape

        infected_strain_cross_protection = infected_strain_params.cross_protection
        cross_protection_strains = list(infected_strain_cross_protection.keys())
        msg = "Strain cross immunity incorrectly specified"
        assert cross_protection_strains == strain_strata, msg

        # ... and its protection against infection with a new index strain.
        for infecting_strain in cross_protection_strains:
            strain_combination_protections = infected_strain_cross_protection[infecting_strain]

            source_filter = {"strain": infected_strain}
            dest_filter = {"strain": infecting_strain}

            # Apply the modification to the early recovered compartment
            cross_protection = 1. - strain_combination_protections.early_reinfection
            adjusters = {
                strat: Multiply(cross_protection * multiplier) for
                strat, multiplier in infection_adjustment.items()
            }

            immunity_strat.set_flow_adjustments(
                FlowName.EARLY_REINFECTION,
                adjusters,
                source_strata=source_filter,
                dest_strata=dest_filter,
            )

            # Apply the immunity-specific protection to the late recovered or "waned" compartment
            cross_protection = 1. - strain_combination_protections.late_reinfection
            adjusters = {
                strat: Multiply(cross_protection * multiplier) for
                strat, multiplier in infection_adjustment.items()
            }
            if "waned" in compartments:
                immunity_strat.set_flow_adjustments(
                    FlowName.LATE_REINFECTION,
                    adjusters,
                    source_strata=source_filter,
                    dest_strata=dest_filter,
                )

    return immunity_strat
