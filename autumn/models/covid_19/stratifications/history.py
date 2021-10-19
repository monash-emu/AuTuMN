from summer import Stratification

from autumn.models.covid_19.parameters import Parameters
from autumn.models.covid_19.constants import COMPARTMENTS, History, HISTORY_STRATA
from autumn.models.covid_19.strat_processing.clinical import (
    add_clinical_adjustments_to_strat, get_all_adjustments, get_blank_adjustments_for_strat,
    update_adjustments_for_strat
)


def get_history_strat(params: Parameters) -> Stratification:
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

    # Severity parameter for previously infected persons
    severity_adjuster_request = params.rel_prop_symptomatic_experienced
    severity_adjuster_experienced = params.rel_prop_symptomatic_experienced if severity_adjuster_request else 1.

    # Add the clinical adjustments parameters as overwrites in a similar way as for vaccination
    adjs = get_all_adjustments(
        params.clinical_stratification, params.country, params.population, params.infection_fatality.props,
        params.sojourn, severity_adjuster_experienced, severity_adjuster_experienced,
        severity_adjuster_experienced, params.infection_fatality.top_bracket_overwrite,
    )
    flow_adjs = get_blank_adjustments_for_strat(History.NAIVE)
    flow_adjs = update_adjustments_for_strat(History.EXPERIENCED, flow_adjs, adjs)
    history_strat = add_clinical_adjustments_to_strat(history_strat, flow_adjs)

    return history_strat
