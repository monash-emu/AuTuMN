from summer import Stratification, Multiply

from autumn.models.sm_sir.parameters import Parameters
from autumn.models.sm_sir.constants import BASE_COMPARTMENTS, IMMUNITY_STRATA, ImmunityStratum, FlowName


def get_immunity_strat(params: Parameters, compartments) -> Stratification:

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

    # Adjust infection flows based on the susceptibility of the age group
    infection_adjustment = {
        ImmunityStratum.NONE: Multiply(1.),
        ImmunityStratum.HIGH: Multiply(1. - params.immunity_stratification.infection_risk_reduction.high),
        ImmunityStratum.LOW: Multiply(1. - params.immunity_stratification.infection_risk_reduction.low)
    }
    immunity_strat.set_flow_adjustments(
        FlowName.INFECTION, infection_adjustment
    )

    return immunity_strat
