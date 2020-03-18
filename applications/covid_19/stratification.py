from autumn.tool_kit.utils import split_parameter


def stratify_by_age(model_to_stratify, age_strata, mixing_matrix, total_pops, model_parameters):
    """
    Stratify model by age
    Note that because the string passed is 'agegroup' rather than 'age', the standard SUMMER demography is not triggered
    """
    age_breakpoints = model_parameters['all_stratifications']['agegroup']
    list_of_starting_pops = [i_pop / sum(total_pops) for i_pop in total_pops]
    starting_props = {i_break: prop for i_break, prop in zip(age_breakpoints, list_of_starting_pops)}
    age_breakpoints = [int(i_break) for i_break in age_strata]
    parameter_splits = \
        split_parameter({}, 'to_infectious', age_strata)
    parameter_splits = \
        split_parameter(parameter_splits, 'infect_death', age_strata)
    parameter_splits = \
        split_parameter(parameter_splits, 'within_infectious', age_strata)
    model_to_stratify.stratify(
        "agegroup",
        age_breakpoints,
        [],
        starting_props,
        mixing_matrix=mixing_matrix,
        adjustment_requests=parameter_splits,
        verbose=False
    )
    return model_to_stratify


def update_parameters(
        working_parameters,
        upstream_strata,
        new_low_parameters,
        new_high_parameters,
        parameter_name_to_adjust,
        overwrite=False
):
    strata_being_implemented = \
        ['low', 'high']
    strata_being_implemented = \
        [stratum + 'W' for stratum in strata_being_implemented] if overwrite else strata_being_implemented
    working_parameters.update(
        {parameter_name_to_adjust + 'Xagegroup_' + i_break:
            {
                strata_being_implemented[0]: prop_1,
                strata_being_implemented[1]: prop_2
            }
            for i_break, prop_1, prop_2 in zip(upstream_strata, new_low_parameters, new_high_parameters)
        }
    )
    return working_parameters


def stratify_by_infectiousness(
        _covid_model,
        model_parameters,
        infectious_compartments,
        case_fatality_rates,
        progression_props
):

    # Replicate within infectious progression rates
    within_infectious_rates = [model_parameters['within_infectious']] * 16

    # Calculate death rates and progression rates
    high_infectious_death_rates = \
        [
            cfr / model_parameters['n_infectious_compartments'] * progression for
            cfr, progression in
            zip(case_fatality_rates, within_infectious_rates)
        ]
    high_infectious_within_infectious_rates = \
        [
            (1. - cfr / model_parameters['n_infectious_compartments']) * progression
            for cfr, progression in
            zip(case_fatality_rates, within_infectious_rates)
        ]

    # Progression to high infectiousness, rather than low
    infectious_adjustments = \
        update_parameters(
            {},
            model_parameters['all_stratifications']['agegroup'],
            [1. - prop for prop in progression_props],
            progression_props,
            'to_infectious'
        )

    # Death rates to apply to the high infectious category
    infectious_adjustments = \
        update_parameters(
            infectious_adjustments,
            model_parameters['all_stratifications']['agegroup'],
            [0.] * 16,
            high_infectious_death_rates,
            'infect_death',
            overwrite=True
        )

    # Non-death progression between infectious compartments towards the recovered compartment
    infectious_adjustments = \
        update_parameters(
            infectious_adjustments,
            model_parameters['all_stratifications']['agegroup'],
            within_infectious_rates,
            high_infectious_within_infectious_rates,
            'within_infectious',
            overwrite=True
        )

    # Stratify the model with the SUMMER stratification function
    _covid_model.stratify(
        'infectiousness',
        ['high', 'low'],
        infectious_compartments,
        infectiousness_adjustments=
        {
            'high': model_parameters['high_infect_multiplier'],
            'low': model_parameters['low_infect_multiplier']
        },
        requested_proportions={},
        adjustment_requests=infectious_adjustments,
        verbose=False
    )
    return _covid_model
