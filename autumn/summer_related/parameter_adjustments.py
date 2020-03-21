
def update_parameters(
        adjusted_strata,
        upstream_stratification,
        upstream_strata,
        parameter_sets,
        parameter_name_to_adjust,
        overwrite=False
):
    """
    Add new parameter sets to a model that differ across a previously implemented model stratification and the
    stratification currently being implemented
    """

    # Indicate that parameters currently being implemented are overwrite parameter if needed
    adjusted_strata = [stratum + 'W' for stratum in adjusted_strata] if overwrite else adjusted_strata

    # Loop over upstream strata
    param_adjustments = {}
    for i_upstratum, upstratum in enumerate(upstream_strata):
        sub_param_to_adjust = parameter_name_to_adjust + 'X' + upstream_stratification + '_' + upstratum
        param_adjustments[sub_param_to_adjust] = {}

        # Loop over strata currently being implemented
        for i_stratum, stratum in enumerate(adjusted_strata):
            param_adjustments[sub_param_to_adjust].update(
                {stratum: parameter_sets[i_stratum][i_upstratum]}
            )
    return param_adjustments
