from summer import Stratification

from autumn.models.covid_19.constants import INFECTIOUS_COMPARTMENTS
from autumn.models.covid_19.parameters import Parameters


def get_tracing_strat(params: Parameters) -> Stratification:
    tracing_strat = Stratification("tracing", ["traced", "untraced"], INFECTIOUS_COMPARTMENTS)
    tracing_strat.set_population_split(
        {
            "traced": 0.,
            "untraced": 1.,
        }
    )
    return tracing_strat
