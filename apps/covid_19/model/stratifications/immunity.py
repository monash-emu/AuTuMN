from summer2 import Stratification
from apps.covid_19.constants import COMPARTMENTS
from apps.covid_19.model.parameters import Parameters


IMMUNITY_STRATA = [
    "unvaccinated",
    "vaccinated",
]


def get_immunity_strat(params: Parameters) -> Stratification:
    immunity_strat = Stratification("immunity", IMMUNITY_STRATA, COMPARTMENTS)

    # Everyone starts out unvaccinated
    immunity_strat.set_population_split({"unvaccinated": 1.0, "vaccinated": 0.0})

    return immunity_strat
