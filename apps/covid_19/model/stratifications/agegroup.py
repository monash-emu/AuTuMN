from typing import List

from summer import Multiply, Stratification

from apps.covid_19.constants import COMPARTMENTS
from apps.covid_19.model import preprocess
from apps.covid_19.model.parameters import Parameters
from autumn import inputs
from autumn.utils.utils import normalise_sequence

# Age groups match the Prem matrices
AGEGROUP_STRATA = [
    "0",
    "5",
    "10",
    "15",
    "20",
    "25",
    "30",
    "35",
    "40",
    "45",
    "50",
    "55",
    "60",
    "65",
    "70",
    "75",
]


def get_agegroup_strat(params: Parameters, total_pops: List[int]) -> Stratification:
    """
    Age stratification
    """
    # We use "Stratification" instead of "AgeStratification" for this model, to avoid triggering
    # automatic demography features (which work on the assumption that the time unit is years, so would be totally wrong)
    age_strat = Stratification("agegroup", AGEGROUP_STRATA, COMPARTMENTS)
    country = params.country

    # Dynamic heterogeneous mixing by age
    if params.elderly_mixing_reduction and not params.mobility.age_mixing:
        # Apply eldery protection to the age mixing parameters
        params.mobility.age_mixing = preprocess.elderly_protection.get_elderly_protection_mixing(
            params.elderly_mixing_reduction
        )

    static_mixing_matrix = inputs.get_country_mixing_matrix("all_locations", country.iso3)
    dynamic_mixing_matrix = preprocess.mixing_matrix.build_dynamic_mixing_matrix(
        static_mixing_matrix,
        params.mobility,
        country,
    )

    age_strat.set_mixing_matrix(dynamic_mixing_matrix)

    # Set distribution of starting population
    age_split_props = {
        agegroup: prop for agegroup, prop in zip(AGEGROUP_STRATA, normalise_sequence(total_pops))
    }
    age_strat.set_population_split(age_split_props)

    # Adjust flows based on age group.
    age_strat.add_flow_adjustments(
        "infection", {s: Multiply(v) for s, v in params.age_stratification.susceptibility.items()}
    )
    return age_strat
