
import collections
import autumn.tool_kit as tool_kit


def get_autumn_params():
    """
    Collate all the parameters and the groups to put them into.

    Returns:
        A single dictionary with key params for the individual parameters and key param_groups for the classification
    """

    params = collections.OrderedDict()

    ''' booleans '''

    # collate the boolean keys
    bool_keys \
        = ['output_flow_diagram', 'output_compartment_populations', 'output_riskgroup_fractions',
           'output_age_fractions', 'output_by_subgroups', 'output_fractions', 'output_scaleups', 'output_gtb_plots',
           'output_plot_economics', 'output_plot_riskgroup_checks', 'output_param_plots', 'output_popsize_plot',
           'output_likelihood_plot', 'output_uncertainty', 'write_uncertainty_outcome_params', 'output_spreadsheets',
           'output_documents', 'output_by_scenario', 'output_horizontally', 'output_age_calculations',
           'riskgroup_diabetes', 'riskgroup_hiv', 'riskgroup_prison', 'riskgroup_indigenous', 'riskgroup_urbanpoor',
           'riskgroup_ruralpoor', 'is_lowquality', 'is_amplification', 'is_misassignment', 'is_vary_detection_by_organ',
           'is_timevariant_organs', 'is_treatment_history', 'is_vary_force_infection_by_riskgroup',
           'is_vary_detection_by_riskgroup']
    for i in range(1, 15):
        bool_keys.append('scenario_' + str(i))
    for key in bool_keys:
        params[key] \
            = {'value': False,
               'type': 'boolean',
               'label': tool_kit.find_button_name_from_string(key)}

    # set some boolean keys to on (True) by default
    default_boolean_keys = [
        # 'output_uncertainty',
        'write_uncertainty_outcome_params',
        'output_param_plots',
        # 'is_amplification',
        # 'is_misassignment',
        # 'is_lowquality',
        # 'output_riskgroup_fractions',
        'is_vary_detection_by_organ',
        # 'is_vary_detection_by_riskgroup',
        'is_treatment_history',
        # 'riskgroup_prison',
        'output_likelihood_plot',
        # 'riskgroup_urbanpoor',
        'output_scaleups',
        # 'output_by_subgroups',
        # 'riskgroup_ruralpoor',
        'output_gtb_plots',
        # 'is_vary_force_infection_by_riskgroup',
        'riskgroup_diabetes',
        # 'riskgroup_hiv',
        # 'riskgroup_indigenous',
        # 'is_timevariant_organs'
    ]
    for k in default_boolean_keys:
        params[k]['value'] = True

    ''' drop down and sliders '''

    # countries
    country_options \
        = ['Afghanistan', 'Albania', 'Angola', 'Argentina', 'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'Bahrain',
           'Bangladesh', 'Belarus', 'Belgium', 'Benin', 'Bhutan', 'Botswana', 'Brazil', 'Bulgaria', 'Burundi',
           'Cameroon', 'Chad', 'Chile', 'Croatia', 'Djibouti', 'Ecuador', 'Estonia', 'Ethiopia', 'Fiji', 'Gabon',
           'Georgia', 'Ghana', 'Guatemala', 'Guinea', 'Philippines', 'Romania']
    params['country'] \
        = {'type': 'drop_down',
           'options': country_options,
           'value': 'Fiji'}

    # methodology
    integration_options \
        = ['Runge Kutta', 'Explicit']
    params['integration_method'] \
        = {'type': 'drop_down',
           'options': integration_options,
           'value': integration_options[1]}
    fitting_options \
        = ['Method 1', 'Method 2', 'Method 3', 'Method 4', 'Method 5']
    params['fitting_method'] \
        = {'type': 'drop_down',
           'options': fitting_options,
           'value': fitting_options[-1]}
    params['default_smoothness'] \
        = {'type': 'slider',
           'label': 'Default fitting smoothness',
           'value': 1.,
           'interval': 0.1,
           'min': 0.,
           'max': 5.}
    params['time_step'] \
        = {'type': 'slider',
           'label': 'Integration time step',
           'value': 0.5,
           'min': 0.005,
           'max': 0.5,
           'interval': 0.005}

    # model stratification
    organ_options = ['Pos / Neg / Extra', 'Pos / Neg', 'Unstratified']
    params['organ_strata'] \
        = {'type': 'drop_down',
           'options': organ_options,
           'value': organ_options[0]}
    strain_options = ['Single strain', 'DS / MDR', 'DS / MDR / XDR']
    params['strains'] \
        = {'type': 'drop_down',
           'options': strain_options,
           'value': strain_options[0]}

    # uncertainty options
    uncertainty_options \
        = ['Scenario analysis', 'Epidemiological uncertainty', 'Intervention uncertainty', 'Optimisation (unavailable)',
           'Increment comorbidity']
    params['run_mode'] \
        = {'type': 'drop_down',
           'options': uncertainty_options,
           'value': uncertainty_options[0]}
    params['uncertainty_runs'] \
        = {'type': 'integer',
           'value': 2,
           'label': 'Number of uncertainty runs'}
    params['burn_in_runs'] \
        = {'type': 'integer',
           'value': 0,
           'label': 'Number of burn-in runs'}
    params['search_width'] \
        = {'type': 'double',
           'value': 0.05,
           'label': 'Relative search width'}
    saving_options = ['No saving or loading', 'Load', 'Save']
    params['pickle_uncertainty'] \
        = {'type': 'drop_down',
           'options': saving_options,
           'value': saving_options[0]}
    available_uncertainty_interventions \
        = ['int_prop_treatment_support_relative', 'int_prop_decentralisation', 'int_prop_xpert', 'int_prop_ipt',
           'int_prop_acf', 'int_prop_awareness_raising', 'int_perc_shortcourse_mdr', 'int_perc_firstline_dst',
           'int_perc_treatment_support_relative_ds', 'int_perc_dots_contributor', 'int_perc_dots_groupcontributor']
    params['uncertainty_interventions'] \
        = {'type': 'drop_down',
           'options': available_uncertainty_interventions,
           'value': available_uncertainty_interventions[0]}

    # increment comorbidity
    comorbidity_types = ['Diabetes']
    params['comorbidity_to_increment'] \
        = {'type': 'drop_down',
           'options': comorbidity_types,
           'value': comorbidity_types[0]}

    # set a default label for the key if none has been specified
    for key, value in params.items():
        if not value.get('label'):
            value['label'] = key

    ''' parameter groupings '''

    # initialise the groups
    param_group_keys \
        = ['Model running', 'Model Stratifications', 'Elaborations', 'Scenarios to run', 'Epidemiological uncertainty',
           'Intervention uncertainty', 'Comorbidity incrementing', 'Plotting', 'MS Office outputs']
    param_groups = []
    for group in param_group_keys:
        param_groups.append({'keys': [], 'name': group})

    # distribute the boolean checkbox options
    for key in bool_keys:
        name = params[key]['label']
        if 'Plot' in name or 'Draw' in name:
            param_groups[7]['keys'].append(key)
        elif 'riskgroup_' in key or key[:2] == 'n_':
            param_groups[1]['keys'].append(key)
        elif 'is_' in key:
            param_groups[2]['keys'].append(key)
        elif 'scenario_' in key:
            param_groups[3]['keys'].append(key)
        elif 'uncertainty' in name or 'uncertainty' in key:
            param_groups[4]['keys'].append(key)
        else:
            param_groups[8]['keys'].append(key)

    # distribute other inputs
    for k in ['run_mode', 'country', 'integration_method', 'fitting_method', 'default_smoothness', 'time_step']:
        param_groups[0]['keys'].append(k)
    for k in ['organ_strata', 'strains']:
        param_groups[1]['keys'].append(k)
    for k in ['uncertainty_runs', 'burn_in_runs', 'search_width', 'pickle_uncertainty']:
        param_groups[4]['keys'].append(k)
    for k in ['uncertainty_interventions']:
        param_groups[5]['keys'].append(k)
    for k in ['comorbidity_to_increment']:
        param_groups[6]['keys'].append(k)

    # return single data structure with parameters and parameter groupings
    return {'params': params,
            'param_groups': param_groups}


def convert_params_to_inputs(params):
    """
    Collates all the inputs once the user has hit Run.

    Args:
        params: The set of parameters specified by the user
    Returns:
        Unprocessed inputs for use by the inputs module
    """

    return {key: param['value'] for key, param in params.iteritems()}

