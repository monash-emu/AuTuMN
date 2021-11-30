from typing import Dict

from summer import Stratification

from autumn.models.covid_19.parameters import Parameters
from autumn.models.covid_19.constants import (
    AGE_CLINICAL_TRANSITIONS, PROGRESS, COMPARTMENTS, History, HISTORY_STRATA
)
from autumn.models.covid_19.strat_processing.clinical import (
    add_clinical_adjustments_to_strat, get_all_adjustments, get_blank_adjustments_for_strat,
    update_adjustments_for_strat
)


def get_history_strat(
        params: Parameters, voc_ifr_effects: Dict[str, float], stratified_adjusters: Dict[str, Dict[str, float]],
) -> Stratification:
    """
    Stratification to represent status regarding past infection/disease with Covid.

    Note that we also have a recovered compartment, so people who have previously recovered are actually retained
    within the 'naive' stratum until their immunity wanes, when they transition to treatment experienced.
    For this reason, calculations of the proportion of the population recovered are a little more complicated - see
    request_recovered_outputs in the history.py of the outputs folder.
    """

    history_strat = Stratification("history", HISTORY_STRATA, COMPARTMENTS)

    # Everyone starts out infection-naive
    pop_split = {History.NAIVE: 1., History.EXPERIENCED: 0.}
    history_strat.set_population_split(pop_split)

    # Add the clinical adjustments parameters as overwrites in a similar way as for vaccination
    flow_adjs = {}
    for voc in voc_ifr_effects.keys():

        adjs = get_all_adjustments(
            params.clinical_stratification, params.country, params.population, params.infection_fatality.props,
            params.sojourn, stratified_adjusters[voc]["ifr"], stratified_adjusters[voc]["sympt"],
            stratified_adjusters[voc]["hosp"]
        )

        # Get them into the format needed to be applied to the model
        flow_adjs[voc] = get_blank_adjustments_for_strat([PROGRESS, *AGE_CLINICAL_TRANSITIONS])
        update_adjustments_for_strat(History.EXPERIENCED, flow_adjs, adjs, voc)

    add_clinical_adjustments_to_strat(history_strat, flow_adjs, History.NAIVE, list(voc_ifr_effects.keys()))

    return history_strat
