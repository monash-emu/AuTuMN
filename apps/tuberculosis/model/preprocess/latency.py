from summer.model.utils.parameter_processing import (
    create_step_function_from_dict,
    get_parameter_dict_from_function,
)

from autumn.tool_kit import add_w_to_param_names, change_parameter_unit


# get parameter values from Ragonnet et al., Epidemics 2017
def get_unstratified_parameter_values(params):
    for param_name in ['stabilisation_rate', 'early_activation_rate', 'late_activation_rate']:
        params[param_name] = 365.25 * params['age_specific_latency'][param_name]['unstratified']
    return params


def get_adapted_age_parameters(age_breakpoints, age_specific_latency):
    """
    Get age-specific latency parameters adapted to any specification of age breakpoints
    """
    adapted_parameter_dict = {}
    for parameter in ("early_activation_rate", "stabilisation_rate", "late_activation_rate"):
        del age_specific_latency[parameter]['unstratified']
        adapted_parameter_dict[parameter] = add_w_to_param_names(
            change_parameter_unit(
                get_parameter_dict_from_function(
                    create_step_function_from_dict(age_specific_latency[parameter]),
                    age_breakpoints,
                ),
                365.251,
            )
        )
    return adapted_parameter_dict


def edit_adjustments_for_diabetes(adjustments, age_breakpoints, prop_diabetes, diabetes_age_start,
                                  rr_progression_diabetes):
    raw_diabetes_multiplier = prop_diabetes * rr_progression_diabetes + 1. - prop_diabetes
    for i, age_breakpoint in enumerate(age_breakpoints):
        if i == len(age_breakpoints) - 1:
            multiplier = raw_diabetes_multiplier
        elif age_breakpoints[i+1] <= diabetes_age_start:
            multiplier = 1.
        else:
            range_above = age_breakpoints[i+1] - diabetes_age_start
            full_range = age_breakpoints[i+1] - age_breakpoint
            range_above = min(range_above, full_range)
            multiplier = (range_above * raw_diabetes_multiplier + (full_range - range_above)) / full_range
        for stage in ['early', 'late']:
            param_name = stage + '_activation_rate'
            adjustments[param_name][str(age_breakpoint) + "W"] *= multiplier
    return adjustments
