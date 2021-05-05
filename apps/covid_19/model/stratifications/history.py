from summer import Stratification

from apps.covid_19.model.parameters import Parameters
from apps.covid_19.constants import COMPARTMENTS
from apps.covid_19.model.preprocess.vaccination import add_clinical_adjustments_to_strat

HISTORY_STRATA = [
    "naive",
    "experienced",
]


def get_history_strat(params: Parameters) -> Stratification:
    """
    Stratification to represent status regarding past infection/disease with Covid.
    """
    history_strat = Stratification(
        "history",
        HISTORY_STRATA,
        COMPARTMENTS,
    )

    # Everyone starts out infection-naive.
    history_strat.set_population_split({"naive": 1., "experienced": 0.})

    # Placeholder parameters for the effect of past infection on protection against severe disease given infection.
    history_severity_effect = 1.

    # Add the clinical adjustments parameters as overwrites in the same way as for vaccination.
    history_strat = add_clinical_adjustments_to_strat(
        history_strat,
        HISTORY_STRATA[0],
        HISTORY_STRATA[1],
        params,
        history_severity_effect,
        history_severity_effect,
        history_severity_effect
    )

    return history_strat
