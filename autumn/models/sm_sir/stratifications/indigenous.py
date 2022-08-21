import pandas as pd
from typing import List

from summer import Stratification

from autumn.core.inputs.demography.queries import get_population_by_agegroup


def get_indigenous_strat(
    compartments: List[str],
    overall_indigenous_prop: float,
) -> Stratification:
    """
    Get the Indigenous stratification object.

    Args:
        compartments: All the model compartments
        overall_indigenous_prop: The proportion of the total population indigenous
    Returns:
        The Indigenous stratification object
    """

    strat = Stratification("indigenous", ["indigenous", "non_indigenous"], compartments)
    strat.set_population_split(
        {
            "indigenous": overall_indigenous_prop, 
            "non_indigenous": 1. - overall_indigenous_prop
        }
    )
    return strat
