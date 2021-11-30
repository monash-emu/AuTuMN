from typing import Dict

from summer import Multiply, Stratification

from autumn.models.covid_19.parameters import Parameters
from autumn.models.covid_19.constants import COMPARTMENTS, History, HISTORY_STRATA, INFECTION, DISEASE_COMPARTMENTS
from autumn.models.covid_19.stratifications.vaccination import apply_immunity_to_strat


def get_history_strat(params: Parameters, stratified_adjusters: Dict[str, Dict[str, float]]) -> Stratification:
    """
    Stratification to represent status regarding past infection/disease with Covid.
    Currently three strata, with everyone entering the experienced stratum after they have recovered from an episode.

    Args:
        params:
        voc_ifr_effects:
        stratified_adjusters:

    Returns:
        The history stratification summer object for application to the main model

    """

    history_strat = Stratification("history", HISTORY_STRATA, COMPARTMENTS)

    # Everyone starts out infection-naive
    pop_split = {History.NAIVE: 1., History.EXPERIENCED: 0., History.WANED: 0.}
    history_strat.set_population_split(pop_split)

    history_params = params.history
    change_strata = HISTORY_STRATA[1:]  # The affected strata are all but the first, which is the unvaccinated

    history_effects = apply_immunity_to_strat(history_strat, params, stratified_adjusters, History.NAIVE)

    # Vaccination effect against infection
    infect_adjs = {strat: Multiply(1. - history_effects[strat]["infection_efficacy"]) for strat in change_strata}
    infect_adjs.update({History.NAIVE: None})
    history_strat.add_flow_adjustments(INFECTION, infect_adjs)

    # Vaccination effect against infectiousness
    infectious_adjs = {s: Multiply(1. - getattr(getattr(history_params, s), "ve_infectiousness")) for s in change_strata}
    infectious_adjs.update({History.NAIVE: None})
    for compartment in DISEASE_COMPARTMENTS:
        history_strat.add_infectiousness_adjustments(compartment, infectious_adjs)

    return history_strat
