from typing import List
import numpy as np

from summer import Stratification, Multiply

from autumn.models.covid_19.mixing_matrix import build_dynamic_mixing_matrix
from autumn.models.sm_sir.parameters import Parameters
from autumn.models.sm_sir.constants import FlowName
from autumn.tools.utils.utils import normalise_sequence


def get_agegroup_strat(
        params: Parameters, total_pops: List[int], mixing_matrices: np.array, compartments: List[str],
        is_dynamic_matrix: bool,
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
        mixing_matrices: The static age-specific mixing matrix
        compartments: All the model compartments
        is_dynamic_matrix: Whether to use the dynamically scaling matrix or the static (all locations) mixing matrix

    Returns:
        The age stratification summer object

    """

    age_strat = Stratification("agegroup", params.age_groups, compartments)

    # Heterogeneous mixing by age
    final_matrix = build_dynamic_mixing_matrix(mixing_matrices, params.mobility, params.country) if is_dynamic_matrix \
        else mixing_matrices["all_locations"]
    age_strat.set_mixing_matrix(final_matrix)

    # Set distribution of starting population
    age_split_props = {str(agegroup): prop for agegroup, prop in zip(params.age_groups, normalise_sequence(total_pops))}
    age_strat.set_population_split(age_split_props)

    # Adjust infection flows based on the susceptibility of the age group
    age_suscept = params.age_stratification.susceptibility
    age_strat.set_flow_adjustments(FlowName.INFECTION, {sus: Multiply(value) for sus, value in age_suscept.items()})

    return age_strat
