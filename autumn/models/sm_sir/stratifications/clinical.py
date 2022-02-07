from typing import List, Union
from copy import copy

from summer import Stratification, Multiply, Overwrite, CompartmentalModel
from summer.compute import ComputedValueProcessor

from autumn.models.sm_sir.constants import ClinicalStratum, Compartment
from autumn.models.sm_sir.strat_processing.clinical import get_cdr_func
from autumn.models.sm_sir.parameters import Parameters


class FunctionWrapper(ComputedValueProcessor):
    """
    Very basic processor that wraps a time/computed values function
    of the type used in flow and adjusters

    FIXME:
    This is such a basic use case, it probably belongs in summer

    """

    def __init__(self, function_to_wrap: callable):
        """
        Initialise with just the param function
        Args:
            function_to_wrap: The function
        """

        self.wrapped_function = function_to_wrap

    def process(self, compartment_values, computed_values, time):
        return self.wrapped_function(time, computed_values)


def get_clinical_strat(
        model: CompartmentalModel,
        compartments: List[str],
        params: Parameters,
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
        params: All model parameters
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

    # Prepare for including incomplete detection
    if is_undetected:
        clinical_strata = [ClinicalStratum.SYMPT_NON_DETECT] + clinical_strata  # "Pre-pending"
        cdr_func, non_detect_func = get_cdr_func(detect_prop, params)
        model.add_computed_value_process("cdr", FunctionWrapper(cdr_func))
        model.add_computed_value_process("undetected_prop", FunctionWrapper(non_detect_func))

    # Prepare for including asymptomatic cases
    if sympt_props:
        clinical_strata = [ClinicalStratum.ASYMPT] + clinical_strata  # Pre-pending again

    # Create the stratification object
    clinical_strat = Stratification("clinical", clinical_strata, comps_to_stratify)

    # Implement the splitting for symptomatic/asymptomatic status
    if sympt_props:
        for i_age, age_group in enumerate(age_groups):
            sympt_prop = sympt_props[i_age]
            asympt_prop = 1.0 - sympt_prop

            if is_undetected:

                def abs_cdr_func(time, computed_values, age_sympt_prop=sympt_prop):
                    return computed_values["cdr"] * age_sympt_prop

                def abs_non_detect_func(time, computed_values, age_sympt_prop=sympt_prop):
                    return computed_values["undetected_prop"] * age_sympt_prop

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

    # Infectiousness adjustments
    asympt_effect = params.asympt_infectiousness_effect
    isolate_effect = params.isolate_infectiousness_effect

    # Start from blank adjustments
    base_infectiousness = {stratum: None for stratum in clinical_strata}

    # Account for asymptomatics being less infectious, if they are included in the model
    if sympt_props:
        base_infectiousness.update({ClinicalStratum.ASYMPT: Overwrite(asympt_effect)})

    # Add in the effect of isolation if partial case detection is being simulated, otherwise only asymptomatic effect
    isolate_infectiousness = copy(base_infectiousness)
    if is_undetected:
        isolate_infectiousness.update({ClinicalStratum.DETECT: Overwrite(isolate_effect)})

    # Apply the isolation adjustments (which might be the same as the base) to the last infectious compartment
    clinical_strat.add_infectiousness_adjustments(comps_to_stratify[-1], isolate_infectiousness)

    # If there are two infectious compartments, apply the adjustment without isolation to the first one
    if Compartment.INFECTIOUS_LATE in comps_to_stratify:
        clinical_strat.add_infectiousness_adjustments(Compartment.INFECTIOUS, base_infectiousness)

    return clinical_strat
