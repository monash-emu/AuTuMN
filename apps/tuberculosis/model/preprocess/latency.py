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
