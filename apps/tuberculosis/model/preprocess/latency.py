from summer.model.utils.parameter_processing import (
    create_step_function_from_dict,
    get_parameter_dict_from_function,
)

from autumn.tool_kit import add_w_to_param_names, change_parameter_unit


# get parameter values from Ragonnet et al., Epidemics 2017
def get_unstratified_parameter_values(params):
    params["stabilisation_rate"] = 365.25 * 1.0e-2
    params["early_activation_rate"] = 365.25 * 1.1e-3
    params["late_activation_rate"] = 365.25 * 5.5e-6

    return params


def get_adapted_age_parameters(age_breakpoints):
    """
    Get age-specific latency parameters adapted to any specification of age breakpoints
    """
    adapted_parameter_dict = {}
    for parameter in ("early_activation_rate", "stabilisation_rate", "late_activation_rate"):
        adapted_parameter_dict[parameter] = add_w_to_param_names(
            change_parameter_unit(
                get_parameter_dict_from_function(
                    create_step_function_from_dict(AGE_SPECIFIC_LATENCY_PARAMETERS[parameter]),
                    age_breakpoints,
                ),
                365.251,
            )
        )
    return adapted_parameter_dict


# All the latency progression parameters from Ragonnet et al.
AGE_SPECIFIC_LATENCY_PARAMETERS = {
    "early_activation_rate": {0: 6.6e-3, 5: 2.7e-3, 15: 2.7e-4},
    "stabilisation_rate": {0: 1.2e-2, 5: 1.2e-2, 15: 5.4e-3},
    "late_activation_rate": {0: 1.9e-11, 5: 6.4e-6, 15: 3.3e-6},
}
