from typing import List
from summer import Stratification, Multiply

from autumn.models.sm_covid.constants import IMMUNITY_STRATA, ImmunityStratum, FlowName


def adjust_susceptible_infection_without_strains(
        immune_effect: float,        
        immunity_strat: Stratification,
):
    """
    Apply the modification to the immunity stratification to account for immunity to first infection (from the
    susceptible compartment), i.e. vaccine-induced immunity (or for some models this stratification could be taken
    to represent past infection prior to the beginning of the simulation period).

    Args:
        immune_effect: The protection from vaccination
        immunity_strat: The immunity stratification, to be modified

    """

    infection_adjustments = {
        ImmunityStratum.UNVACCINATED: None,
        ImmunityStratum.VACCINATED: Multiply(1.0 - immune_effect),
    }

    immunity_strat.set_flow_adjustments(
        FlowName.INFECTION,
        infection_adjustments,
    )


def get_immunity_strat(
        compartments: List[str],
) -> Stratification:
    """
    Args:
        compartments: Unstratified model compartment types being implemented

    Returns:
        The summer Stratification object that captures vaccine-related immunity

    """

    # Create the immunity stratification, which applies to all compartments
    immunity_strat = Stratification("immunity", IMMUNITY_STRATA, compartments)

    # Set distribution of starting population
    p_immune = .5
    immunity_split_props = {
        ImmunityStratum.UNVACCINATED: 1.0 - p_immune,
        ImmunityStratum.VACCINATED: p_immune
    }
    immunity_strat.set_population_split(immunity_split_props)

    return immunity_strat
