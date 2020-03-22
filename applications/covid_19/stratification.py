from autumn.tool_kit.utils import split_parameter, find_series_compartment_parameter
from autumn.demography.ageing import add_agegroup_breaks
from applications.covid_19.covid_outputs import create_request_stratified_incidence_covid
from autumn.tool_kit.utils import repeat_list_elements
from autumn.constants import Compartment
from autumn.summer_related.parameter_adjustments import \
    adjust_upstream_stratified_parameter, split_prop_into_two_subprops


def stratify_by_age(model_to_stratify, mixing_matrix, total_pops, model_parameters, output_connections):
    """
    Stratify model by age
    Note that because the string passed is 'agegroup' rather than 'age', the standard automatic SUMMER demography is not
    triggered
    """
    model_parameters = \
        add_agegroup_breaks(model_parameters)
    age_strata = \
        model_parameters['all_stratifications']['agegroup']
    list_of_starting_pops = \
        [i_pop / sum(total_pops) for i_pop in total_pops]
    starting_props = \
        {i_break: prop for i_break, prop in zip(age_strata, list_of_starting_pops)}
    parameter_splits = \
        split_parameter({}, 'to_infectious', age_strata)
    parameter_splits = \
        split_parameter(parameter_splits, 'infect_death', age_strata)
    parameter_splits = \
        split_parameter(parameter_splits, 'within_infectious', age_strata)
    model_to_stratify.stratify(
        'agegroup',
        [int(i_break) for i_break in age_strata],
        [],
        starting_props,
        mixing_matrix=mixing_matrix,
        adjustment_requests=parameter_splits,
        verbose=False
    )
    output_connections.update(
        create_request_stratified_incidence_covid(
            model_parameters['incidence_stratification'],
            model_parameters['all_stratifications'],
            model_parameters['n_compartment_repeats']
        )
    )
    return model_to_stratify, model_parameters, output_connections


def repeat_list_elements_average_last_two(raw_props):
    """
    Repeat 5-year age-specific proportions, but with 75+s taking the average of the last two groups
    """
    repeated_props = repeat_list_elements(2, raw_props[:-1])
    repeated_props[-1] = sum(raw_props[-2:]) / 2.
    return repeated_props


def stratify_by_infectiousness(_covid_model, model_parameters, compartments):
    """
    Stratify the infectious compartments of the covid model (not including the presymptomatic compartments, which are
    actually infectious)
    """

    # New strata names
    # strata_to_implement = \
    #     ['low', 'moderate', 'high']

    strata_to_implement = \
        ['non_infectious', 'infectious_non_hospital', 'hospital_non_icu', 'icu']

    # Find the compartments that will need to be stratified under this stratification
    compartments_to_split = \
        [i_comp for i_comp in compartments if i_comp.startswith(Compartment.INFECTIOUS)]

    # Repeat the 5-year age-specific CFRs for all but the top age bracket, and average the last two for the last group
    case_fatality_rates = \
        repeat_list_elements(2, model_parameters['age_cfr'][: -1]) + \
        [(model_parameters['age_cfr'][-1] + model_parameters['age_cfr'][-2]) / 2.]

    # Repeat all the 5-year age-specific infectiousness proportions, with adjustment for data length as needed
    infectious_props = repeat_list_elements(2, model_parameters['age_infect_progression'])
    hospital_props = repeat_list_elements_average_last_two(model_parameters['hospital_props'])
    icu_props = repeat_list_elements_average_last_two(model_parameters['icu_props'])

    # Find the absolute  progression proportions from the relative splits
    abs_props = split_prop_into_two_subprops([1.] * 16, '', infectious_props, 'infectious')
    abs_props.update(split_prop_into_two_subprops(abs_props['infectious'], 'infectious', hospital_props, 'hospital'))
    abs_props.update(split_prop_into_two_subprops(abs_props['hospital'], 'hospital', icu_props, 'icu'))

    # Replicate within infectious progression rates for all age groups
    within_infectious_rates = [model_parameters['within_infectious']] * 16

    # Calculate death rates and progression rates
    high_infectious_death_rates = \
        [
            find_series_compartment_parameter(cfr, model_parameters['n_compartment_repeats'], progression) for
            cfr, progression in
            zip(case_fatality_rates, within_infectious_rates)
        ]
    high_infectious_within_infectious_rates = \
        [
            find_series_compartment_parameter(1. - cfr, model_parameters['n_compartment_repeats'], progression) for
            cfr, progression in
            zip(case_fatality_rates, within_infectious_rates)
        ]

    # Progression to high infectiousness, rather than low
    infectious_adjustments = {}
    infectious_adjustments.update(
        adjust_upstream_stratified_parameter(
            'to_infectious',
            strata_to_implement,
            'agegroup',
            model_parameters['all_stratifications']['agegroup'],
            [abs_props['non_infectious'],
             abs_props['infectious_non_hospital'],
             abs_props['hospital_non_icu'],
             abs_props['icu']]
        )
    )

    # Death rates to apply to the high infectious category
    # infectious_adjustments.update(
    #     adjust_upstream_stratified_parameter(
    #         'infect_death',
    #         strata_to_implement,
    #         'agegroup',
    #         model_parameters['all_stratifications']['agegroup'],
    #         [[0.] * 16, [0.] * 16, high_infectious_death_rates],
    #         overwrite=True
    #     )
    # )

    # Non-death progression between infectious compartments towards the recovered compartment
    # infectious_adjustments.update(
    #     adjust_upstream_stratified_parameter(
    #         'within_infectious',
    #         strata_to_implement,
    #         'agegroup',
    #         model_parameters['all_stratifications']['agegroup'],
    #         [within_infectious_rates, [0.] * 16, high_infectious_within_infectious_rates],
    #         overwrite=True
    #     )
    # )

    # Stratify the model with the SUMMER stratification function
    _covid_model.stratify(
        'infectiousness',
        strata_to_implement,
        compartments_to_split,
        # infectiousness_adjustments=
        # {
        #     'high': model_parameters['high_infect_multiplier'],
        #     'moderate': model_parameters['low_infect_multiplier'],
        #     'low': model_parameters['low_infect_multiplier']
        # },
        requested_proportions={
            'non_infectious': 1. / 4.,
            'infectious_non_hospital': 1. / 4.,
            'hospital_non_icu': 1. / 4.,
            'icu': 1. / 4.
        },
        adjustment_requests=infectious_adjustments,
        verbose=False
    )
    return _covid_model
