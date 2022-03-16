from typing import List, Union, Tuple, Dict
from copy import copy

from summer import Stratification, Multiply, Overwrite, CompartmentalModel

from autumn.tools.utils.utils import FunctionWrapper
from autumn.models.sm_sir.constants import ClinicalStratum, Compartment
from autumn.models.sm_sir.strat_processing.clinical import get_cdr_func
from autumn.models.sm_sir.parameters import Parameters


def add_detection_processes_to_model(
        detect_prop: float,
        params: Parameters,
        model: CompartmentalModel,
) -> Tuple[callable, callable]:
    """
    Add the case detection processes to the model object. Just to avoid repeating a few lines of code in the main
    conditional of get_clinical_strat.

    Args:
        detect_prop: The default fixed case detection rate
        params: All the submitted parameters
        model: The model to be modified

    Returns:
        The functions representing the proportion detected and the proportion not detected

    """

    cdr_func, non_detect_func = get_cdr_func(detect_prop, params)
    model.add_computed_value_process("cdr", FunctionWrapper(cdr_func))
    model.add_computed_value_process("undetected_prop", FunctionWrapper(non_detect_func))
    return cdr_func, non_detect_func


def get_clinical_strat(
        model: CompartmentalModel,
        compartments: List[str],
        params: Parameters,
        infectious_entry_flow: str,
        detect_prop: float,
        is_detect_split: bool,
        sympt_props: Union[None, float, Dict[str, float]],
) -> Union[None, Stratification]:
    """
    Only stratify the infectious compartments, because in the dynamic model we are only interested in the
    epidemiological effect - and these are the only infectious compartments (unlike the Covid model).
    For the symptomatic fractions - this can only be implemented as age-specific. There is currently no code to allow
    for a constant symptomatic/asymptomatic fraction (although this could be done relatively easily).
    If only partial case detection is applied and not an asymptomatic fraction, then the case detection rate should be
    considered as the proportion of all infections that are detected as cases.

    Args:
        model: The model object we are working with, even though the stratification is applied outside this function
        compartments: Unstratified model compartment types
        params: All model parameters
        infectious_entry_flow: The name of the flow that takes people into the (first) infectious compartment(s)
        detect_prop: Proportion of symptomatic cases detected
        is_detect_split: Whether undetected population is being simulated
        sympt_props: Symptomatic proportions, or None if stratification by symptomatic status not required

    Returns:
        Clinical stratification object to be applied in the main model file, or None if stratification irrelevant

    """

    # Identify the compartment(s) to stratify, one or two depending on whether the infectious compartment is split
    comps_to_stratify = [comp for comp in compartments if "infectious" in comp]

    # Check requests have come through in the correct format
    if type(sympt_props) == dict:

        msg = "Age stratification not applied, which is the only reason for having a dict of sympt props here"
        model_stratifications = [strat.name for strat in model._stratifications]
        assert "agegroup" in model_stratifications, msg

        msg = "Symptomatic proportions do not correspond to the age stratification"
        agegroup_strat_index = model_stratifications.index("agegroup")
        assert list(sympt_props.keys()) == model._stratifications[agegroup_strat_index].strata, msg

    else:
        msg = "Symptomatic split not specified in the correct format"
        assert type(sympt_props) == float, msg

    # Split by symptomatic status and by detection status
    if sympt_props and is_detect_split:

        # Create the stratification object, with all three possible strata
        clinical_strat = Stratification(
            "clinical",
            [ClinicalStratum.ASYMPT, ClinicalStratum.SYMPT_NON_DETECT, ClinicalStratum.DETECT],
            comps_to_stratify
        )
        _, _ = add_detection_processes_to_model(detect_prop, params, model)

        # If the model is age stratified and we have a user request to split according to symptomatic status by age
        if type(sympt_props) == dict:
            for age_group, sympt_prop in sympt_props.items():

                def abs_cdr_func(time, computed_values, age_sympt_prop=sympt_prop):
                    return computed_values["cdr"] * age_sympt_prop

                def abs_non_detect_func(time, computed_values, age_sympt_prop=sympt_prop):
                    return computed_values["undetected_prop"] * age_sympt_prop

                adjustments = {
                    ClinicalStratum.ASYMPT: Multiply(1. - sympt_prop),
                    ClinicalStratum.SYMPT_NON_DETECT: Multiply(abs_non_detect_func),
                    ClinicalStratum.DETECT: Multiply(abs_cdr_func),
                }
                clinical_strat.set_flow_adjustments(
                    infectious_entry_flow,
                    adjustments,
                    dest_strata={"agegroup": age_group}
                )

        else:

            def abs_cdr_func(time, computed_values, sympt_prop=sympt_props):
                return computed_values["cdr"] * sympt_prop

            def abs_non_detect_func(time, computed_values, sympt_prop=sympt_props):
                return computed_values["undetected_prop"] * sympt_prop

            adjustments = {
                ClinicalStratum.ASYMPT: Multiply(1. - sympt_props),
                ClinicalStratum.SYMPT_NON_DETECT: Multiply(abs_non_detect_func),
                ClinicalStratum.DETECT: Multiply(abs_cdr_func),
            }
            clinical_strat.set_flow_adjustments(
                infectious_entry_flow,
                adjustments,
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

        # If the model is age stratified and we have a dictionary for splitting according to symptomatic status by age
        if type(sympt_props) == dict:
            for age_group, sympt_prop in sympt_props.items():
                adjustments = {
                    ClinicalStratum.ASYMPT: Multiply(1. - sympt_prop),
                    ClinicalStratum.DETECT: Multiply(sympt_prop),
                }
                clinical_strat.set_flow_adjustments(
                    infectious_entry_flow,
                    adjustments,
                    dest_strata={"agegroup": age_group}
                )

        # Otherwise if we just have a single value for the symptomatic proportion
        else:
            adjustments = {
                ClinicalStratum.ASYMPT: Multiply(1. - sympt_props),
                ClinicalStratum.DETECT: Multiply(sympt_props),
            }
            clinical_strat.set_flow_adjustments(
                infectious_entry_flow,
                adjustments,
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
        non_detect_func, cdr_func = add_detection_processes_to_model(detect_prop, params, model)
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
