import pandas as pd
from typing import List

from summer import Stratification

from autumn.core.inputs.demography.queries import get_population_by_agegroup


def get_indigenous_strat(
    compartments: List[str],
) -> Stratification:
    """
    Get the Indigenous stratification object.

    Args:
        compartments: All the model compartments
    Returns:
        The Indigenous stratification object
    """

    strat = Stratification("indigenous", ["indigenous", "non_indigenous"], compartments)

    return strat
