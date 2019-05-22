
import collections
import autumn.tool_kit as tool_kit
import six


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
        = ['output_flow_diagram', 'output_compartment_populations', 'output_age_fractions', 'output_by_subgroups',
           'output_scaleups', 'output_epi_plots', 'output_plot_economics', 'output_param_plots',
           'output_likelihood_plot', 'write_uncertainty_outcome_params', 'output_spreadsheets', 'output_documents',
           'output_by_scenario', 'output_horizontally',
           'riskgroup_diabetes', 'riskgroup_hiv', 'riskgroup_prison', 'riskgroup_indigenous', 'riskgroup_urbanpoor',
           'riskgroup_ruralpoor', 'riskgroup_dorm', 'is_lowquality', 'is_amplification', 'is_misassignment', 'is_vary_detection_by_organ',
           'is_timevariant_organs', 'is_treatment_history', 'is_vary_force_infection_by_riskgroup',
           'is_vary_detection_by_riskgroup', 'is_include_relapse_in_ds_outcomes', 'is_include_hiv_treatment_outcomes',
           'is_adjust_population', 'is_shortcourse_improves_outcomes', 'plot_option_vars_two_panels',
           'plot_option_overlay_input_data', 'plot_option_title', 'plot_option_plot_all_vars',
           'plot_option_overlay_gtb', 'plot_option_overlay_targets']
    for i in range(1, 15):
        bool_keys.append('scenario_' + str(i))
    for key in bool_keys:
        params[key] \
            = {'value': False,
               'type': 'boolean',
               'label': tool_kit.find_title_from_dictionary(key)}

    ''' drop down and sliders '''

    # countries
    country_options \
        = ['Afghanistan', 'Albania', 'Angola', 'Argentina', 'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'Bahrain',
           'Bangladesh', 'Belarus', 'Belgium', 'Benin', 'Bhutan', 'Botswana', 'Brazil', 'Bulgaria', 'Burundi',
           'Cameroon', 'Chad', 'Chile', 'Croatia', 'Djibouti', 'Ecuador', 'Estonia', 'Ethiopia', 'Fiji', 'Gabon',
           'Georgia', 'Ghana', 'Guatemala', 'Guinea', 'Philippines', 'Romania']
    params['country'] \
        = {'type': 'drop_down',
           'options': country_options}

    # methodology
    integration_options \
        = ['Runge Kutta', 'Explicit']
    params['integration_method'] \
        = {'type': 'drop_down',
           'options': integration_options}
    fitting_options \
        = ['Method 1', 'Method 2', 'Method 3', 'Method 4', 'Method 5']
    params['fitting_method'] \
        = {'type': 'drop_down',
           'options': fitting_options}
    params['default_smoothness'] \
        = {'type': 'slider',
           'label': 'Default fitting smoothness',
           'value': 1.,
           'interval': .1,
           'min': 0.,
           'max': 5.}
    params['time_step'] \
        = {'type': 'slider',
           'label': 'Integration time step',
           'value': .5,
           'min': 5e-3,
           'max': .5,
           'interval': 5e-3}

    # model stratification
    organ_options = ['Pos / Neg / Extra', 'Pos / Neg', 'Unstratified']
    params['organ_strata'] \
        = {'type': 'drop_down',
           'options': organ_options}
    strain_options = ['Single strain', 'DS / MDR', 'DS / MDR / XDR']
    params['strains'] \
        = {'type': 'drop_down',
           'options': strain_options}

    # uncertainty options
    uncertainty_options \
        = ['Scenario analysis', 'Epidemiological uncertainty', 'Intervention uncertainty', 'Optimisation (unavailable)',
           'Increment comorbidity', 'Rapid calibration']
    params['run_mode'] \
        = {'type': 'drop_down',
           'options': uncertainty_options}
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
           'value': 2e-1,
           'label': 'Relative search width'}
    saving_options = ['No saving or loading', 'Load', 'Save', 'Store in DB', 'Load from DB']
    params['pickle_uncertainty'] \
        = {'type': 'drop_down',
           'options': saving_options}
    available_uncertainty_intervention \
        = ['int_prop_treatment_support_relative', 'int_prop_decentralisation', 'int_prop_xpert', 'int_prop_ipt',
           'int_prop_acf', 'int_prop_awareness_raising', 'int_perc_shortcourse_mdr', 'int_perc_firstline_dst',
           'int_perc_treatment_support_relative_ds', 'int_perc_dots_contributor', 'int_perc_dots_groupcontributor']
    params['uncertainty_intervention'] \
        = {'type': 'drop_down',
           'options': available_uncertainty_intervention}
    params['n_centiles_for_shading'] \
        = {'type': 'integer',
           'value': 100,
           'min': 5}
    params['n_samples'] \
        = {'type': 'integer',
           'value': 20}
    params['plot_option_start_time'] \
        = {'type': 'double',
           'value': 1990.}
    params['plot_option_end_time'] \
        = {'type': 'double',
           'value': 2020.}

    # increment comorbidity
    comorbidity_types = ['Diabetes']
    params['comorbidity_to_increment'] \
        = {'type': 'drop_down',
           'options': comorbidity_types}

    # set a default label for the key if none has been specified
    for key, value in params.items():
        if not value.get('label'):
            value['label'] = key

    # set some boolean keys to on (True) by default
    default_boolean_keys = [
        'is_adjust_population',
        'plot_option_vars_two_panels',
        'plot_option_overlay_input_data',
        'plot_option_title',
        'plot_option_overlay_gtb',
        'plot_option_overlay_targets',
        'write_uncertainty_outcome_params',
        # 'output_param_plots',
        # 'output_likelihood_plot',
        'is_shortcourse_improves_outcomes',
        # 'plot_option_plot_all_vars',
        'is_amplification',
        'is_misassignment',
        # 'is_lowquality',
        'is_vary_detection_by_organ',
        #'is_include_relapse_in_ds_outcomes',
        'is_vary_detection_by_riskgroup',
        # 'is_include_hiv_treatment_outcomes',
        'is_treatment_history',
        # 'riskgroup_prison',
        # 'riskgroup_urbanpoor',
        'output_scaleups',
        'output_by_subgroups',
        'output_compartment_populations',
        # 'riskgroup_ruralpoor',
        'output_epi_plots',
        'is_vary_force_infection_by_riskgroup',   # heterpgeneous mixing
        # 'riskgroup_diabetes',
        'riskgroup_dorm'
        # 'riskgroup_hiv',
        # 'riskgroup_indigenous',
        # 'is_timevariant_organs'
        # 'output_plot_economics'
    ]
    for k in default_boolean_keys:
        params[k]['value'] = True

    # set default values for drop down lists
    for param in params:
        if params[param]['type'] == 'drop_down':
            params[param]['value'] = params[param]['options'][0]
    params['fitting_method']['value'] = params['fitting_method']['options'][-1]
    params['integration_method']['value'] = params['integration_method']['options'][1]
    params['strains']['value'] = params['strains']['options'][1]
    params['country']['value'] = 'Bhutan'

    ''' parameter groupings '''

    # initialise the groups
    param_group_keys \
        = ['Model running', 'Model Stratifications', 'Elaborations', 'Scenarios to run', 'Epidemiological uncertainty',
           'Intervention uncertainty', 'Comorbidity incrementing', 'Plots', 'Plot options', 'MS Office outputs']
    param_groups = []
    for group in param_group_keys:
        param_groups.append({'keys': [], 'name': group})
    for tab in [5, 6, 8]:
        param_groups[tab]['attr'] = {'webgui': True}

    # distribute the boolean checkbox options
    for key in bool_keys:
        name = params[key]['label']
        if 'plot_option' in key:
            param_groups[8]['keys'].append(key)
        elif 'Plot' in name or 'Draw' in name:
            param_groups[7]['keys'].append(key)
        elif 'riskgroup_' in key or key.startswith('n_'):
            param_groups[1]['keys'].append(key)
        elif 'is_' in key:
            param_groups[2]['keys'].append(key)
        elif 'scenario_' in key:
            param_groups[3]['keys'].append(key)
        elif 'uncertainty' in name or 'uncertainty' in key:
            param_groups[4]['keys'].append(key)
        else:
            param_groups[9]['keys'].append(key)

    # distribute other inputs
    for k in ['run_mode', 'country', 'integration_method', 'fitting_method', 'default_smoothness', 'time_step']:
        param_groups[0]['keys'].append(k)
    for k in ['organ_strata', 'strains']:
        param_groups[1]['keys'].append(k)
    for k in ['uncertainty_runs', 'burn_in_runs', 'search_width', 'pickle_uncertainty', 'n_centiles_for_shading']:
        param_groups[4]['keys'].append(k)
    for k in ['uncertainty_intervention', 'n_samples']:
        param_groups[5]['keys'].append(k)
    for k in ['comorbidity_to_increment']:
        param_groups[6]['keys'].append(k)
    for k in ['plot_option_start_time', 'plot_option_end_time']:
        param_groups[8]['keys'].append(k)

    params['age_breakpoints'] \
        = {'type': 'breakpoints',
           'label': 'Age Breakpoints',
           'value': [5, 15, 25]}
    param_groups[1]['keys'].append('age_breakpoints')

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

    # replacing iteritems with items for py3
    return {key: param['value'] for key, param in params.items()}

