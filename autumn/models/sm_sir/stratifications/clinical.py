from typing import List, Union
from copy import copy

from summer import Stratification, Multiply, Overwrite, CompartmentalModel

from autumn.tools.utils.utils import FunctionWrapper
from autumn.models.sm_sir.constants import ClinicalStratum, Compartment
from autumn.models.sm_sir.strat_processing.clinical import get_cdr_func
from autumn.models.sm_sir.parameters import Parameters


def get_clinical_strat(
        model: CompartmentalModel,
        compartments: List[str],
        params: Parameters,
        age_groups: List[int],
        infectious_entry_flow: str,
        detect_prop: float,
        is_detect_split: bool,
        sympt_props: Union[None, List[float]],
) -> Union[None, Stratification]:
    """
    Only stratify the infectious compartments, because in the dynamic model we are only interested in the
    epidemiological effect - and these are the only infectious compartments (unlike the Covid model).
    We can only get into this function, if either incomplete case detection or asymptomatic people are being included in
    the model.
    For the symptomatic fractions - this can only be implemented as age-specific. There is currently no code to allow
    for a constant symptomatic/asymptomatic fraction (although this could be done relatively easily).
    If only partial case detection is applied and not an asymptomatic fraction, then the case detection rate should be
    considered as the proportion of all infections that are detected as cases.

    Args:
        model: The model object we are working with, even though the stratification is applied outside this function
        compartments: Unstratified model compartment types
        params: All model parameters
        age_groups: Modelled age groups
        infectious_entry_flow: The name of the flow that takes people into the (first) infectious compartment(s)
        detect_prop: Proportion of symptomatic cases detected
        is_detect_split: Whether undetected population is being simulated
        sympt_props: Symptomatic proportions, or None if stratification by symptomatic status not required

    Returns:
        Clinical stratification object to be applied in the main model file, or None if stratification irrelevant

    """

    # Identify the compartment(s) to stratify, one or two depending on whether the infectious compartment is split
    comps_to_stratify = [comp for comp in compartments if "infectious" in comp]

    # Split by symptomatic status and by detection status
    if sympt_props and is_detect_split:

        # Create the stratification object, with all three possible strata
        clinical_strat = Stratification(
            "clinical",
            [ClinicalStratum.ASYMPT, ClinicalStratum.SYMPT_NON_DETECT, ClinicalStratum.DETECT],
            comps_to_stratify
        )

        # Work out the splits based on symptomatic status and detection
        for i_age, age_group in enumerate(age_groups):
            sympt_prop = sympt_props[i_age]
            asympt_prop = 1. - sympt_prop

            def abs_cdr_func(time, computed_values, age_sympt_prop=sympt_prop):
                return computed_values["cdr"] * age_sympt_prop

            def abs_non_detect_func(time, computed_values, age_sympt_prop=sympt_prop):
                return computed_values["undetected_prop"] * age_sympt_prop

            cdr_func, non_detect_func = get_cdr_func(detect_prop, params)
            model.add_computed_value_process("cdr", FunctionWrapper(cdr_func))
            model.add_computed_value_process("undetected_prop", FunctionWrapper(non_detect_func))
            adjustments = {
                ClinicalStratum.ASYMPT: Multiply(asympt_prop),
                ClinicalStratum.SYMPT_NON_DETECT: Multiply(abs_non_detect_func),
                ClinicalStratum.DETECT: Multiply(abs_cdr_func),
            }
            clinical_strat.set_flow_adjustments(
                infectious_entry_flow,
                adjustments,
                dest_strata={"agegroup": str(age_group)}
            )

        # Work out the infectiousness adjustments
        base_infectiousness = {stratum: None for stratum in clinical_strat.strata}
        base_infectiousness.update({ClinicalStratum.ASYMPT: Overwrite(params.asympt_infectiousness_effect)})
        isolate_infectiousness = copy(base_infectiousness)
        isolate_infectiousness.update({ClinicalStratum.DETECT: Overwrite(params.isolate_infectiousness_effect)})

    # Only apply the symptomatic split, which differs by age group
    elif sympt_props and not is_detect_split:

        # Create the stratification object, with just asymptomatic and symptomatic (called detect)
        clinical_strat = Stratification(
            "clinical",
            [ClinicalStratum.ASYMPT, ClinicalStratum.DETECT],
            comps_to_stratify
        )

        # Work out the splits based on symptomatic status
        for i_age, age_group in enumerate(age_groups):
            sympt_prop = sympt_props[i_age]
            asympt_prop = 1. - sympt_prop
            adjustments = {
                ClinicalStratum.ASYMPT: Multiply(asympt_prop),
                ClinicalStratum.DETECT: Multiply(sympt_prop),
            }
            clinical_strat.set_flow_adjustments(
                infectious_entry_flow,
                adjustments,
                dest_strata={"agegroup": str(age_group)}
            )

        # Work out the infectiousness adjustments
        base_infectiousness = {stratum: None for stratum in clinical_strat.strata}
        base_infectiousness.update({ClinicalStratum.ASYMPT: Overwrite(params.asympt_infectiousness_effect)})
        isolate_infectiousness = copy(base_infectiousness)

    # Only stratify by detection status, which applies equally to all age groups
    elif not sympt_props and is_detect_split:

        # Create the stratification object, with detected and undetected
        clinical_strat = Stratification(
            "clinical",
            [ClinicalStratum.SYMPT_NON_DETECT, ClinicalStratum.DETECT],
            comps_to_stratify
        )

        # Work out the splits based on detection
        cdr_func, non_detect_func = get_cdr_func(detect_prop, params)
        model.add_computed_value_process("cdr", FunctionWrapper(cdr_func))
        model.add_computed_value_process("undetected_prop", FunctionWrapper(non_detect_func))
        adjustments = {
            ClinicalStratum.SYMPT_NON_DETECT: Multiply(non_detect_func),
            ClinicalStratum.DETECT: Multiply(cdr_func),
        }
        clinical_strat.set_flow_adjustments(
            infectious_entry_flow,
            adjustments,
        )

        # Work out the infectiousness adjustments
        base_infectiousness = {stratum: None for stratum in clinical_strat.strata}
        isolate_infectiousness = copy(base_infectiousness)
        isolate_infectiousness.update({ClinicalStratum.DETECT: Overwrite(params.isolate_infectiousness_effect)})

    # No stratification to apply if neither of the two processes are implemented
    else:
        return None

    # Apply the isolation adjustments to the last infectious compartment (which could be infectious or late infectious)
    clinical_strat.add_infectiousness_adjustments(comps_to_stratify[-1], isolate_infectiousness)

    # If two infectious compartments, apply the adjustment without isolation to the early one, called infectious
    if Compartment.INFECTIOUS_LATE in comps_to_stratify:
        clinical_strat.add_infectiousness_adjustments(Compartment.INFECTIOUS, base_infectiousness)

    return clinical_strat
