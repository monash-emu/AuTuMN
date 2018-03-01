
import collections
import autumn.tool_kit as tool_kit


def find_button_name_from_string(working_string):
    """
    Find the string to attach to a boolean check box for the GUI, either from a dictionary, or using a specific approach
    for things like scenario names.

    Args:
        working_string: AuTuMN's name for the boolean quantity
    Returns:
        A more user-friendly string for the GUI
    """

    button_name_dictionary \
        = {'output_uncertainty':
               'Run uncertainty',
           'write_uncertainty_outcome_params':
               'Record parameters',
           'output_spreadsheets':
               'Write to spreadsheets',
           'output_documents':
               'Write to documents',
           'output_by_scenario':
               'Output by scenario',
           'output_horizontally':
               'Write horizontally',
           'output_gtb_plots':
               'Plot outcomes',
           'output_compartment_populations':
               'Plot compartment sizes',
           'output_by_subgroups':
               'Plot outcomes by sub-groups',
           'output_age_fractions':
               'Plot proportions by age',
           'output_riskgroup_fractions':
               'Plot proportions by risk group',
           'output_flow_diagram':
               'Draw flow diagram',
           'output_fractions':
               'Plot compartment fractions',
           'output_scaleups':
               'Plot scale-up functions',
           'output_plot_economics':
               'Plot economics graphs',
           'output_plot_riskgroup_checks':
               'Plot risk group checks',
           'output_age_calculations':
               'Plot age calculation weightings',
           'output_param_plots':
               'Plot parameter progression',
           'output_popsize_plot':
               'Plot "popsizes" for cost-coverage curves',
           'output_likelihood_plot':
               'Plot log likelihoods over runs',
           'riskgroup_diabetes':
               'Type II diabetes',
           'riskgroup_hiv':
               'HIV',
           'riskgroup_prison':
               'Prison',
           'riskgroup_urbanpoor':
               'Urban poor',
           'riskgroup_ruralpoor':
               'Rural poor',
           'riskgroup_indigenous':
               'Indigenous',
           'is_lowquality':
               'Low quality care',
           'is_amplification':
               'Resistance amplification',
           'is_timevariant_organs':
               'Time-variant organ status',
           'is_misassignment':
               'Strain mis-assignment',
           'is_vary_detection_by_organ':
               'Vary case detection by organ status',
           'n_organs':
               'Number of organ strata',
           'n_strains':
               'Number of strains',
           'is_vary_force_infection_by_riskgroup':
               'Heterogeneous mixing',
           'is_treatment_history':
               'Treatment history'}

    if working_string in button_name_dictionary:
        return button_name_dictionary[working_string]
    elif 'scenario_' in working_string:
        return tool_kit.capitalise_first_letter(tool_kit.replace_underscore_with_space(working_string))
    else:
        return working_string


def get_autumn_params():
    """
    Collate all the parameters and the groups to put them into.

    Returns:
        A single dictionary with key params for the individual parameters and key param_groups for the classification
    """

    params = collections.OrderedDict()

    # collate the boolean keys
    bool_keys \
        = ['output_flow_diagram', 'output_compartment_populations', 'output_riskgroup_fractions',
           'output_age_fractions', 'output_by_subgroups', 'output_fractions', 'output_scaleups', 'output_gtb_plots',
           'output_plot_economics', 'output_plot_riskgroup_checks', 'output_param_plots', 'output_popsize_plot',
           'output_likelihood_plot', 'output_uncertainty', 'write_uncertainty_outcome_params', 'output_spreadsheets',
           'output_documents', 'output_by_scenario', 'output_horizontally', 'output_age_calculations',
           'riskgroup_diabetes', 'riskgroup_hiv', 'riskgroup_prison', 'riskgroup_indigenous', 'riskgroup_urbanpoor',
           'riskgroup_ruralpoor', 'is_lowquality', 'is_amplification', 'is_misassignment', 'is_vary_detection_by_organ',
           'is_timevariant_organs', 'is_treatment_history', 'is_vary_force_infection_by_riskgroup']
    for i in range(1, 15):
        bool_keys.append('scenario_' + str(i))
    for key in bool_keys:
        params[key] \
            = {'value': False,
               'type': 'boolean',
               'label': find_button_name_from_string(key)}

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
           'value': 1.0,
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
    params['n_organs'] \
        = {'type': 'drop_down',
           'options': organ_options,
           'value': organ_options[0]}
    strain_options = ['Single strain', 'DS / MDR', 'DS / MDR / XDR']
    params['n_strains'] \
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

    # set a default label for the key if none has been specified
    for key, value in params.items():
        if not value.get('label'):
            value['label'] = key

    ''' parameter groupings '''

    # initialise the groups
    param_group_keys \
        = ['Model running', 'Model Stratifications', 'Elaborations', 'Scenarios to run', 'Uncertainty', 'Plotting',
           'MS Office outputs']
    param_groups = []
    for group in param_group_keys:
        param_groups.append({'keys': [], 'name': group})

    # distribute the boolean checkbox options
    for key in bool_keys:
        name = params[key]['label']
        if 'Plot' in name or 'Draw' in name:
            param_groups[5]['keys'].append(key)
        elif 'riskgroup_' in key or key[:2] == 'n_':
            param_groups[1]['keys'].append(key)
        elif 'is_' in key:
            param_groups[2]['keys'].append(key)
        elif 'scenario_' in key:
            param_groups[3]['keys'].append(key)
        elif 'uncertainty' in name or 'uncertainty' in key:
            param_groups[4]['keys'].append(key)
        else:
            param_groups[6]['keys'].append(key)

    # distribute other inputs
    for k in ['run_mode', 'country', 'integration_method', 'fitting_method', 'default_smoothness', 'time_step']:
        param_groups[0]['keys'].append(k)
    for k in ['n_organs', 'n_strains']:
        param_groups[1]['keys'].append(k)
    for k in ['uncertainty_runs', 'burn_in_runs', 'search_width', 'pickle_uncertainty']:
        param_groups[4]['keys'].append(k)

    # return single data structure with parameters and parameter groupings
    return {'params': params,
            'param_groups': param_groups}


def convert_params_to_inputs(params):
    """
    Collates all the inputs once the user has hit Run.

    Args:
        params: The set of parameters specified by the user
    Returns:
        inputs: Unprocessed inputs for use by the inputs module
    """

    # keys for drop-down lists to be converted to integers
    organ_stratification_keys \
        = {'Unstratified': 0,
           'Pos / Neg': 2,
           'Pos / Neg / Extra': 3}
    strain_stratification_keys \
        = {'Single strain': 0,
           'DS / MDR': 2,
           'DS / MDR / XDR': 3}

    # starting inputs always includes the baseline scenario
    inputs = {'scenarios_to_run': [0], 'scenario_names_to_run': ['baseline']}

    # add all of the user inputs
    for key, param in params.iteritems():
        value = param['value']
        if param['type'] == 'boolean':
            if 'scenario_' not in key:
                inputs[key] = param['value']
            elif param['value']:
                i_scenario = int(key[9:])
                inputs['scenarios_to_run'].append(i_scenario)
                inputs['scenario_names_to_run'].append(tool_kit.find_scenario_string_from_number(i_scenario))
        elif param['type'] == 'drop_down':
            if key == 'fitting_method':
                inputs[key] = int(value[-1])
            elif key == 'n_organs':
                inputs[key] = organ_stratification_keys[value]
            elif key == 'n_strains':
                inputs[key] = strain_stratification_keys[value]
            else:
                inputs[key] = value
        else:
            inputs[key] = value

    return inputs

