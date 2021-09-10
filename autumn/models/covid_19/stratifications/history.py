from summer import Stratification

from autumn.models.covid_19.parameters import Parameters
from autumn.models.covid_19.constants import COMPARTMENTS, History, HISTORY_STRATA
from autumn.models.covid_19.preprocess.vaccination import add_clinical_adjustments_to_strat


def get_history_strat(params: Parameters) -> Stratification:
    """
    Stratification to represent status regarding past infection/disease with Covid.
    """
    history_strat = Stratification(
        "history",
        HISTORY_STRATA,
        COMPARTMENTS,
    )

    # Everyone starts out infection-naive
    history_strat.set_population_split({History.NAIVE: 1., History.EXPERIENCED: 0.})

    # Severity parameter for previously infected persons
    rel_prop_symptomatic_experienced = (
        params.rel_prop_symptomatic_experienced if params.rel_prop_symptomatic_experienced else 1.
    )

    # Add the clinical adjustments parameters as overwrites in the same way as for vaccination
    history_strat = add_clinical_adjustments_to_strat(
        history_strat,
        History.NAIVE,
        History.EXPERIENCED,
        params,
        rel_prop_symptomatic_experienced,
        rel_prop_symptomatic_experienced,
        rel_prop_symptomatic_experienced,
        params.infection_fatality.top_bracket_overwrite,
    )

    return history_strat
