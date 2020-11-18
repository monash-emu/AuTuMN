from summer.model.utils.parameter_processing import (
    create_step_function_from_dict,
    get_parameter_dict_from_function,
)

from autumn.tool_kit import add_w_to_param_names, change_parameter_unit
from autumn.curve import tanh_based_scaleup, make_linear_curve


# get parameter values from Ragonnet et al., Epidemics 2017
def get_unstratified_parameter_values(params):
    for param_name in ["stabilisation_rate", "early_activation_rate", "late_activation_rate"]:
        params[param_name] = (
            365.25 * params["age_specific_latency"][param_name]["unstratified"]
            if "age" not in params["stratify_by"]
            else 1.0
        )
    return params


def get_adapted_age_parameters(age_breakpoints, age_specific_latency):
    """
    Get age-specific latency parameters adapted to any specification of age breakpoints
    """
    adapted_parameter_dict = {}
    for parameter in ("early_activation_rate", "stabilisation_rate", "late_activation_rate"):
        del age_specific_latency[parameter]["unstratified"]
        for age_break in ["0", "5", "15"]:
            old_key = "age_" + age_break
            new_key = int(age_break)
            age_specific_latency[parameter][new_key] = age_specific_latency[parameter].pop(old_key)
        adapted_parameter_dict[parameter] = change_parameter_unit(
            get_parameter_dict_from_function(
                create_step_function_from_dict(age_specific_latency[parameter]),
                age_breakpoints,
            ),
            365.251,
        )

    return adapted_parameter_dict


def edit_adjustments_for_diabetes(
    model, adjustments, age_breakpoints, prop_diabetes, rr_progression_diabetes, future_diabetes_multiplier
):
    diabetes_scale_up = tanh_based_scaleup(b=0.05, c=1980, sigma=0.0, upper_asymptote=1.0)
    future_diabetes_trend = make_linear_curve(x_0=2020, x_1=2050, y_0=1, y_1=future_diabetes_multiplier)

    def combined_diabetes_scale_up(t):
        multiplier = 1.
        if t > 2020:
            multiplier = future_diabetes_trend(t)
        return multiplier * diabetes_scale_up(t)

    for i, age_breakpoint in enumerate(age_breakpoints):
        for stage in ["early", "late"]:
            param_name = stage + "_activation_rate"
            stratified_param_name = param_name + "Xage_" + str(age_breakpoint)
            function_name = stratified_param_name + "_func"
            unadjusted_progression_rate = adjustments[param_name][str(age_breakpoint)]
            adjustments[param_name][str(age_breakpoint)] = function_name
            model.time_variants[function_name] = make_age_diabetes_scaleup_func(
                prop_diabetes[age_breakpoint],
                combined_diabetes_scale_up,
                rr_progression_diabetes,
                unadjusted_progression_rate,
            )
            model.parameters[stratified_param_name] = stratified_param_name
    return adjustments


def make_age_diabetes_scaleup_func(
    final_prop_diabetes, scale_up_func, rr_progression_diabetes, original_param_val
):
    def diabetes_func(t):
        return (
            1.0 - scale_up_func(t) * final_prop_diabetes * (1.0 - rr_progression_diabetes)
        ) * original_param_val

    return diabetes_func
