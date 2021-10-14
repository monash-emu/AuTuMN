from summer import Stratification

from autumn.models.covid_19.parameters import Parameters
from autumn.models.covid_19.constants import COMPARTMENTS, History, HISTORY_STRATA
from autumn.models.covid_19.preprocess.clinical import add_clinical_adjustments_to_strat


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

    # Add the clinical adjustments parameters as overwrites in the same way as for vaccination
    history_strat = add_clinical_adjustments_to_strat(
        history_strat, History.NAIVE, History.EXPERIENCED, params, severity_adjuster_experienced,
        severity_adjuster_experienced, severity_adjuster_experienced, params.infection_fatality.top_bracket_overwrite,
    )

    return history_strat
