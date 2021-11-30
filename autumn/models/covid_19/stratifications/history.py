from typing import Dict

from summer import Stratification, Overwrite

from autumn.models.covid_19.parameters import Parameters
from autumn.models.covid_19.constants import (
    AGE_CLINICAL_TRANSITIONS, PROGRESS, COMPARTMENTS, History, HISTORY_STRATA, INFECTION
)
from autumn.models.covid_19.strat_processing.clinical import (
    add_clinical_adjustments_to_strat, get_all_adjustments, get_blank_adjustments_for_strat,
    update_adjustments_for_strat
)
from autumn.models.covid_19.strat_processing.vaccination import get_stratum_vacc_history_effect


def get_history_strat(
        params: Parameters, voc_ifr_effects: Dict[str, float], stratified_adjusters: Dict[str, Dict[str, float]],
) -> Stratification:
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

    history_effects, flow_adjs = {}, {}
    modified_strata = HISTORY_STRATA[1:]  # The affected strata are all but the first, which is the unvaccinated

    # Add the clinical adjustments parameters as overwrites in a similar way as for vaccination
    vocs = list(stratified_adjusters.keys())
    for voc in vocs:
        flow_adjs[voc] = get_blank_adjustments_for_strat([PROGRESS, *AGE_CLINICAL_TRANSITIONS])
        for stratum in modified_strata:

            # Collate the vaccination effects together
            strat_args = (params, stratum, stratified_adjusters[voc], "history")
            history_effects[stratum], sympt_adj, hosp_adj, ifr_adj = get_stratum_vacc_history_effect(*strat_args)

            # Get the adjustments by clinical status and age group applicable to this VoC and vaccination stratum
            adjs = get_all_adjustments(
                params.clinical_stratification, params.country, params.population, params.infection_fatality.props,
                params.sojourn, ifr_adj, sympt_adj, hosp_adj
            )

            # Get into the format needed and apply to both the experienced and waned strata
            update_adjustments_for_strat(stratum, flow_adjs, adjs, voc)
    add_clinical_adjustments_to_strat(history_strat, flow_adjs, History.NAIVE, list(voc_ifr_effects.keys()))

    # Currently set reinfection to zero for the period of time until immunity has waned to retain previous behaviour
    infection_adjs = {History.NAIVE: None, History.EXPERIENCED: Overwrite(0.), History.WANED: None}
    history_strat.add_flow_adjustments(INFECTION, infection_adjs)

    return history_strat
