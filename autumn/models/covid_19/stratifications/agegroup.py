from typing import List, Dict
import numpy as np

from summer import Multiply, Stratification

from autumn.models.covid_19.constants import COMPARTMENTS, AGEGROUP_STRATA, INFECTION
from autumn.models.covid_19.mixing_matrix import build_dynamic_mixing_matrix
from autumn.models.covid_19.parameters import Parameters


def get_agegroup_strat(
        params: Parameters, total_pops: List[int], mixing_matrices: Dict[str, np.ndarray]
) -> Stratification:
    """
    Function to create the age group stratification object.

    We use "Stratification" instead of "AgeStratification" for this model, to avoid triggering
    automatic demography features (which work on the assumption that the time is in years, so would be totally wrong)
    This will be revised in future versions of summer, in which model times will be datetime objects rather than AuTuMN
    bespoke data structures.

    Args:
        params: All model parameters
        total_pops: The population distribution by age
        mixing_matrices: The age-specific mixing matrices

    Returns:
        The age stratification summer object

    """

    age_strat = Stratification("agegroup", AGEGROUP_STRATA, COMPARTMENTS)

    # Get dynamic age-specific mixing matrix
    dynamic_mixing_matrix = build_dynamic_mixing_matrix(mixing_matrices, params.mobility, params.country)
    age_strat.set_mixing_matrix(dynamic_mixing_matrix)

    # Set distribution of starting population
    pop_proportions = (i_value / sum(total_pops) for i_value in total_pops)
    age_split_props = {agegroup: prop for agegroup, prop in zip(AGEGROUP_STRATA, pop_proportions)}
    age_strat.set_population_split(age_split_props)

    # Adjust infection flows based on the susceptibility of the age group
    age_strat_suscept = params.age_stratification.susceptibility
    age_strat.set_flow_adjustments(INFECTION, {sus: Multiply(value) for sus, value in age_strat_suscept.items()})

    return age_strat
