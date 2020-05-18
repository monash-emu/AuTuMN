def change_parameters_to_overwrite(parameters):
    """
    Simple function to add a capital W to the end of all the parameters of interest
    """
    return [parameter + "W" for parameter in parameters]


def adjust_upstream_stratified_parameter(
    base_parameter,
    implement_strata,
    upstream_stratification,
    upstream_strata,
    parameter_values,
    overwrite=False,
):
    """
    Add new parameter sets to a model that differ across a previously implemented model stratification and the
    stratification currently being implemented
    """

    # Indicate that parameters currently being implemented are overwrite if requested
    implement_strata = (
        change_parameters_to_overwrite(implement_strata) if overwrite else implement_strata
    )

    # Loop over upstream strata
    param_adjustments = {}
    for i_upstream_stratum, upstream_stratum in enumerate(upstream_strata):
        sub_param_to_adjust = (
            base_parameter + "X" + upstream_stratification + "_" + upstream_stratum
        )
        param_adjustments[sub_param_to_adjust] = {}

        # Loop over strata currently being implemented
        for i_stratum, stratum in enumerate(implement_strata):
            param_adjustments[sub_param_to_adjust].update(
                {stratum: parameter_values[i_stratum][i_upstream_stratum]}
            )
    return param_adjustments


def split_prop_into_two_subprops(overall_props, overall_prop_name, split_props, split_prop_name):
    """
    Split a proportion parameter into two subproportions according to a requested split, and include the complement
    """
    extra_underscore = "" if overall_prop_name == "" else "_"
    return {
        split_prop_name: [
            overall_props[i_agegroup] * split_props[i_agegroup]
            for i_agegroup in range(len(split_props))
        ],
        overall_prop_name
        + extra_underscore
        + "non_"
        + split_prop_name: [
            overall_props[i_agegroup] * (1.0 - split_props[i_agegroup])
            for i_agegroup in range(len(split_props))
        ],
    }


def split_parameter(parameter, strata):
    return {parameter: {stratum: 1.0 for stratum in strata}}


def split_multiple_parameters(parameters, strata):
    adjustments = {}
    for param in parameters:
        adjustments.update(split_parameter(param, strata))
    return adjustments
