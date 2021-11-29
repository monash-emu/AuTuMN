from typing import List, Callable
import numpy as np

from summer import Stratification, Multiply

from autumn.models.sm_sir.parameters import Parameters
from autumn.models.sm_sir.constants import COMPARTMENTS, AGEGROUP_STRATA
from autumn.tools.utils.utils import normalise_sequence


def get_agegroup_strat(
        params: Parameters, total_pops: List[int], mixing_matrix: np.array
) -> Stratification:
    """
    Age group stratification

    We use "Stratification" instead of "AgeStratification" for this model, to avoid triggering
    automatic demography features (which work on the assumption that the time is in years, so would be totally wrong)
    This will be revised in future versions of summer, in which model times will be datetime objects rather than AuTuMN
    bespoke data structures.
    """

    age_strat = Stratification("agegroup", AGEGROUP_STRATA, COMPARTMENTS)

    # Heterogeneous mixing by age
    age_strat.set_mixing_matrix(mixing_matrix)

    # Set distribution of starting population
    age_split_props = {
        agegroup: prop for agegroup, prop in zip(AGEGROUP_STRATA, normalise_sequence(total_pops))
    }
    age_strat.set_population_split(age_split_props)

    # Adjust infection flows based on the susceptibility of the age group
    age_strat.add_flow_adjustments(
        "infection", {sus: Multiply(value) for sus, value in params.age_stratification.susceptibility.items()}
    )
    return age_strat
