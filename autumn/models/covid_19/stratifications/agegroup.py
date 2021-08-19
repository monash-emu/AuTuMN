from typing import List

from summer import Multiply, Stratification

from autumn.models.covid_19.constants import COMPARTMENTS, AGEGROUP_STRATA
from autumn.models.covid_19.model import preprocess
from autumn.models.covid_19.parameters import Parameters
from autumn.tools import inputs
from autumn.tools.utils.utils import normalise_sequence


def get_agegroup_strat(params: Parameters, total_pops: List[int]) -> Stratification:
    """
    Age stratification
    """
    # We use "Stratification" instead of "AgeStratification" for this model, to avoid triggering
    # automatic demography features (which work on the assumption that the time is in years, so would be totally wrong)
    age_strat = Stratification("agegroup", AGEGROUP_STRATA, COMPARTMENTS)
    country = params.country

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
