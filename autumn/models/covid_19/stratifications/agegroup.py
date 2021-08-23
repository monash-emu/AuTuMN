from typing import List

from summer import Multiply, Stratification

from autumn.models.covid_19.constants import COMPARTMENTS, AGEGROUP_STRATA, INFECTION
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

    # Dynamic heterogeneous mixing by age
    if params.age_specific_risk_multiplier and not params.mobility.age_mixing:
        # Apply adjustments to the age mixing parameters
        age_categories = params.age_specific_risk_multiplier.age_categories
        contact_rate_multiplier = params.age_specific_risk_multiplier.contact_rate_multiplier
        if params.age_specific_risk_multiplier.adjustment_start_time:
            adjustment_start_time = params.age_specific_risk_multiplier.adjustment_start_time
        else:
            adjustment_start_time = params.time.start
        if params.age_specific_risk_multiplier.adjustment_end_time:
            adjustment_end_time = params.age_specific_risk_multiplier.adjustment_end_time
        else:
            adjustment_end_time = params.time.end

        params.mobility.age_mixing = preprocess.age_specific_risk.get_adjusted_age_specific_mixing(
            age_categories, adjustment_start_time, adjustment_end_time, contact_rate_multiplier
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

    # Adjust flows based on age group
    age_strat.add_flow_adjustments(
        INFECTION, {sus: Multiply(value) for sus, value in params.age_stratification.susceptibility.items()}
    )
    return age_strat
