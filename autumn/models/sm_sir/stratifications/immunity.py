from typing import List
from summer import Stratification, Multiply

from autumn.models.sm_sir.parameters import Parameters
from autumn.models.sm_sir.constants import IMMUNITY_STRATA, ImmunityStratum, FlowName


def get_immunity_strat(params: Parameters, compartments: List[str], vacc_immune_escape) -> Stratification:

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

    # Consider each strain separately
    for strain in vacc_immune_escape:

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
        immunity_strat.set_flow_adjustments(
            FlowName.INFECTION,
            infection_adjustment,
            dest_strata={"strain": strain},
        )
        immunity_strat.set_flow_adjustments(
            FlowName.EARLY_REINFECTION,
            infection_adjustment,
            dest_strata={"strain": strain},
        )
        immunity_strat.set_flow_adjustments(
            FlowName.LATE_REINFECTION,
            infection_adjustment,
            dest_strata={"strain": strain},
        )

    return immunity_strat
