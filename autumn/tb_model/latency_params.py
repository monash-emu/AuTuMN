"""
Latency parameters from Ragonnet et al.
"""
from summer_py.summer_model.utils.parameter_processing import (
    create_step_function_from_dict,
    get_parameter_dict_from_function,
)

from ..tool_kit import add_w_to_param_names, change_parameter_unit


def manually_create_age_specific_latency_parameters(model_parameters):
    age_specific_latency_parameters = {}
    for parameter in ['early_progression', 'stabilisation', 'late_progression']:
        age_specific_latency_parameters[parameter] = {}
        for age_group in [0, 5, 15]:
            age_specific_latency_parameters[parameter].update(
                {age_group: model_parameters[parameter + '_' + str(age_group)]}
            )
    return age_specific_latency_parameters


def provide_aggregated_latency_parameters():
    """
    function to add the latency parameters estimated by Ragonnet et al from our paper in Epidemics to the existing
    parameter dictionary
    """
    return AGGREGATED_LATENCY_PARAMETERS


def get_adapted_age_parameters(age_breakpoints):
    """
    Get age-specific latency parameters adapted to any specification of age breakpoints
    """
    adapted_parameter_dict = {}
    for parameter in ("early_progression", "stabilisation", "late_progression"):
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


# Aggregated latency parameters estimated by Ragonnet et al from our paper in Epidemics
AGGREGATED_LATENCY_PARAMETERS = {
    "early_progression": 1.1e-3,
    "stabilisation": 1.0e-2,
    "late_progression": 5.5e-6,
}

# All the latency progression parameters from Ragonnet et al.
AGE_SPECIFIC_LATENCY_PARAMETERS = {
    "early_progression": {0: 6.6e-3, 5: 2.7e-3, 15: 2.7e-4},
    "stabilisation": {0: 1.2e-2, 5: 1.2e-2, 15: 5.4e-3},
    "late_progression": {0: 1.9e-11, 5: 6.4e-6, 15: 3.3e-6},
}


def update_transmission_parameters(parameters, compartments_to_update):
    """
    Update parameters with transmission rates for each compartment with altered immunity/sucseptibility to infection
    """
    for compartment in compartments_to_update:
        parameters.update({
            'contact_rate_' + compartment:
                parameters['contact_rate'] * parameters['rr_transmission_' + compartment]
        })
    return parameters
