from typing import List

from summer import Stratification
from autumn.models.sm_sir.parameters import Parameters
from autumn.models.sm_sir.constants import CLINICAL_STRATA


def get_clinical_strat(params: Parameters, compartments: List[str]) -> Stratification:
    """
    Only stratify the infectious compartments, because in the dynamic model we are only interested in the
    epidemiological effect - and these are the only infectious compartments (unlike the Covid model).

    Args:
        params: Model parameters
        compartments: Unstratified model compartment types

    Returns:
        Clinical stratification object to be applied in the main model file

    """

    # Determine compartments to stratify, dependent on whether the infectious compartment is split
    comps_to_stratify = [comp for comp in compartments if "infectious" in comp]


    clinical_strat = Stratification("clinical", CLINICAL_STRATA, comps_to_stratify)

    return clinical_strat
