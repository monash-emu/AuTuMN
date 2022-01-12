from typing import List

from summer import Stratification
from autumn.models.sm_sir.parameters import Parameters


def get_clinical_strat(params: Parameters, compartments: List[str]) -> Stratification:
    """
    Only stratify the infectious compartments, because in the dynamic model we are only interested in the
    epidemiological effect - and these are the only infectious compartments (unlike the Covid model).

    Args:
        params: Model parameters
        compartments: Unstratified model compartment types

    Returns:
        Nothing yet

    """

    # Determine compartments to stratify, dependent on whether the infectious compartment is split
    comps_to_stratify = [comp for comp in compartments if "infectious" in comp]

    clinical_strata = ["asympt", "sympt_non_detect", "detect"]

    clinical_strat = Stratification("clinical", clinical_strata, comps_to_stratify)

    return clinical_strat
