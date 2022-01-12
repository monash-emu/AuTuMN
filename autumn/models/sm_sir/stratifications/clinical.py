from typing import List, Union

from summer import Stratification, Multiply
from autumn.models.sm_sir.constants import ClinicalStratum


def get_clinical_strat(
        compartments: List[str],
        age_groups: List[int],
        infectious_entry_flow: str,
        detect_prop: float,
        is_undetected: bool,
        sympt_props: Union[None, List[float]],
) -> Stratification:
    """
    Only stratify the infectious compartments, because in the dynamic model we are only interested in the
    epidemiological effect - and these are the only infectious compartments (unlike the Covid model).

    Args:
        compartments: Unstratified model compartment types
        age_groups: Modelled age groups
        infectious_entry_flow: The name of the flow that takes people into the (first) infectious compartment(s)
        detect_prop: Proportion of symptomatic cases detected
        is_undetected: Whether undetected population is being simulated
        sympt_props: Symptomatic proportions, or None if stratification by symptomatic status not required

    Returns:
        Clinical stratification object to be applied in the main model file

    """

    # Identify the compartment(s) to stratify, one or two depending on whether the infectious compartment is split
    comps_to_stratify = [comp for comp in compartments if "infectious" in comp]

    # Start with the two symptomatic strata
    clinical_strata = [ClinicalStratum.DETECT]

    # Work out which strata are to be implemented
    if is_undetected:
        clinical_strata = [ClinicalStratum.SYMPT_NON_DETECT] + clinical_strata

        def cdr_func(time):
            return detect_prop

        def non_detect_func(time):
            return 1.0 - abs_cdr_func(time)

    if sympt_props:
        clinical_strata = [ClinicalStratum.ASYMPT] + clinical_strata

    # Create the stratification object
    clinical_strat = Stratification("clinical", clinical_strata, comps_to_stratify)

    # Implement the splitting for symptomatic/asymptomatic status
    if sympt_props:
        for i_age, age_group in enumerate(age_groups):
            sympt_prop = sympt_props[i_age]
            asympt_prop = 1.0 - sympt_prop

            if is_undetected:

                def abs_cdr_func(time):
                    return cdr_func(time) * sympt_prop

                def abs_non_detect_func(time):
                    return non_detect_func(time) * sympt_prop

                adjustments = {
                    ClinicalStratum.ASYMPT: Multiply(asympt_prop),
                    ClinicalStratum.SYMPT_NON_DETECT: Multiply(abs_non_detect_func),
                    ClinicalStratum.DETECT: Multiply(abs_cdr_func),
                }
            else:
                adjustments = {
                    ClinicalStratum.ASYMPT: Multiply(asympt_prop),
                    ClinicalStratum.DETECT: Multiply(sympt_prop),
                }
            clinical_strat.set_flow_adjustments(
                infectious_entry_flow,
                adjustments,
                dest_strata={"agegroup": str(age_group)}
            )

    # No need for loop over age if symptomatic status not included
    else:

        adjustments = {
            ClinicalStratum.SYMPT_NON_DETECT: Multiply(non_detect_func),
            ClinicalStratum.DETECT: Multiply(cdr_func),
        }
        clinical_strat.set_flow_adjustments(
            infectious_entry_flow,
            adjustments,
        )

    return clinical_strat
