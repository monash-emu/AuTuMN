from typing import Dict

from summer import Stratification

from autumn.models.covid_19.parameters import Parameters
from autumn.models.covid_19.constants import COMPARTMENTS, History, HISTORY_STRATA
from autumn.models.covid_19.stratifications.vaccination import apply_immunity_to_strat


def get_history_strat(params: Parameters, stratified_adjusters: Dict[str, Dict[str, float]]) -> Stratification:
    """
    Stratification to represent status regarding past infection/disease with Covid.
    Currently three strata, with everyone entering the experienced stratum after they have recovered from an episode.

    Args:
        params: All model parameters
        stratified_adjusters: VoC and severity stratification adjusters

    Returns:
        The history stratification summer object for application to the main model

    """

    history_strat = Stratification("history", HISTORY_STRATA, COMPARTMENTS)

    # Everyone starts out infection-naive
    pop_split = {stratum: 0. for stratum in HISTORY_STRATA}
    pop_split[History.NAIVE] = 1.
    history_strat.set_population_split(pop_split)

    # Immunity adjustments equivalent to vaccination approach
    apply_immunity_to_strat(history_strat, params, stratified_adjusters, History.NAIVE)

    return history_strat
