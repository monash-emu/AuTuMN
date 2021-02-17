from summer2 import Stratification, Multiply
from apps.covid_19.constants import COMPARTMENTS
from apps.covid_19.model.parameters import Parameters

from apps.covid_19.model.preprocess.clinical import get_absolute_death_proportions
from apps.covid_19.model.stratifications.clinical import get_ifr_props, get_sympt_props, get_relative_death_props

IMMUNITY_STRATA = [
    "unvaccinated",
    "vaccinated",
]


def get_immunity_strat(params: Parameters) -> Stratification:
    immunity_strat = Stratification("immunity", IMMUNITY_STRATA, COMPARTMENTS)

    # Everyone starts out unvaccinated
    immunity_strat.set_population_split(
        {
            "unvaccinated": 1.0,
            "vaccinated": 0.0
        }
    )

    if params.vaccination:
        immunity_strat.add_flow_adjustments(
            "infection", {
                "vaccinated": Multiply(1. - params.vaccination.efficacy),
                "unvaccinated": None,
            }
        )


    modifier = 0.5
    clinical_params = params.clinical_stratification

    # Get raw IFR and symptomatic proportions
    infection_fatality_props = get_ifr_props(params)
    abs_props = get_sympt_props(params)

    # Get the proportion of people who die for each strata/agegroup, relative to total infected.
    abs_death_props = get_absolute_death_proportions(
        abs_props=abs_props,
        infection_fatality_props=infection_fatality_props,
        icu_mortality_prop=clinical_params.icu_mortality_prop,
    )

    relative_death_props = get_relative_death_props(abs_props, abs_death_props)


    return immunity_strat
