"""
handler.py
=========

RPC-JSON API Web-handlers

To use define functions with name
- public_* public handlers
- login_* requires login first
- admin_* requires admin login

All handlers must return a JSON dictionary, except for downloadable functions
Handlers can take parameters, which are expected to be only JSON-compatible
Python data structures.
"""

from __future__ import print_function

import os
import sys
import json

from flask import session, abort
from flask.ext.login import current_user, login_user, logout_user

import dbmodel


# User handlers

def admin_retrieve_users():
    return {'users': map(dbmodel.parse_user, dbmodel.load_users())}


def public_create_user(user_attr):
    username = dbmodel.check_user_attr(user_attr)['username']
    try:
        dbmodel.load_user(username=username)
    except:
        print(">> public_create_user user_attr", user_attr)
        created_user_attr = dbmodel.create_user(user_attr)
        return {
            'success': True,
            'user': created_user_attr
        }
    else:
        abort(409)


def public_retrieve_current_user():
    return dbmodel.parse_user(current_user)


def login_update_user(user_attr):
    return {
        'success': True,
        'user': dbmodel.update_user_from_attr(user_attr)
    }


def public_login_user(user_attr):
    if not dbmodel.is_current_user_anonymous():
        print(">> public_login_user already logged-in")
        return {
            'success': True,
            'user': dbmodel.parse_user(current_user)
        }

    user_attr = dbmodel.check_user_attr(user_attr)
    kwargs = {}
    if user_attr['username']:
        kwargs['username'] = user_attr['username']
    if user_attr['email']:
        kwargs['email'] = user_attr['email']
    print(">> public_login_user loading", kwargs)

    try:
        user = dbmodel.load_user(**kwargs)
    except:
        pass
    else:
        if user.check_password(user_attr['password']):
            login_user(user)
            return {
                'success': True,
                'user': dbmodel.parse_user(user)
            }

    abort(401)


def admin_delete_user(user_id):
    username = dbmodel.delete_user(user_id)['username']
    print(">> admin_delete_user " + username)


def public_logout_user():
    logout_user()
    session.clear()


# model handlers

bgui_output_lines = []
is_bgui_running = False

sys.path.insert(0, os.path.abspath("../.."))
import autumn.model_runner
import autumn.outputs


def find_scenario_string_from_number(scenario):
    if (scenario is not None):
        return 'baseline'
    else:
        return 'scenario_' + scenario


def convert_params_to_model_inputs(params):
    organ_stratification_keys = {
        'Pos / Neg / Extra': 3,
        'Pos / Neg': 2,
        'Unstratified': 0
    }

    strain_stratification_keys = {
        'Single strain': 0,
        'DS / MDR': 2,
        'DS / MDR / XDR': 3
    }

    model_inputs = {
        'scenarios_to_run': [None],
        'scenario_names_to_run': ['baseline']
    }

    for key, value in params.items():
        if 'scenario_' in key:
            if (value['type'] == 'boolean'):
                if (value['value']):
                    i = int(key[9:11])
                    model_inputs['scenarios_to_run'].append(i)
                    model_inputs['scenario_names_to_run'].append(find_scenario_string_from_number(i))

        elif (key == 'fitting_method'):
            model_inputs[key] = int(value['value'][-1])
        elif (key == 'n_organs'):
            model_inputs[key] = organ_stratification_keys[value['value']]
        elif (key == 'n_strains'):
            model_inputs[key] = strain_stratification_keys[value['value']]
        else:
            model_inputs[key] = value['value']

    return model_inputs


def public_check_autumn_run():
    global bgui_output_lines
    global is_bgui_running
    result = {
        "console": bgui_output_lines,
        "is_running": is_bgui_running
    }
    return result


def bgui_model_output(output_type, data={}):
    if output_type == "init":
        pass
    elif output_type == "console":
        global bgui_output_lines
        bgui_output_lines.append(data["message"])
        print(">> handler.bgui_model_output console:", data["message"])
    elif output_type == "uncertainty_graph":
        with open('graph.json', 'wt') as f:
            f.write(json.dumps(data, indent=2))
        print(">> handler.bgui_model_output uncertainty_graph:", data)


def public_run_autumn(params):
    """
    Run the model
    """
    global is_bgui_running
    global bgui_output_lines
    is_bgui_running = True
    model_inputs = convert_params_to_model_inputs(params)
    print(">> handler.public_run_autumn", json.dumps(model_inputs, indent=2))
    bgui_output_lines = []
    autumn_dir = os.path.join(os.path.dirname(autumn.__file__), os.pardir)
    os.chdir(autumn_dir)
    try:
        model_runner = autumn.model_runner.ModelRunner(
            model_inputs, None, None, bgui_model_output)
        model_runner.master_runner()
        project = autumn.outputs.Project(model_runner, model_inputs)
        project.master_outputs_runner()
        result = {'success': True}
    except:
        result = {'success': False}
    is_bgui_running = False
    return result


def public_get_autumn_params():
    bool_keys = [
        'output_flow_diagram', 'output_compartment_populations', 'output_riskgroup_fractions',
        'output_age_fractions', 'output_by_subgroups', 'output_fractions', 'output_scaleups',
        'output_gtb_plots', 'output_plot_economics', 'output_plot_riskgroup_checks',
        'output_param_plots', 'output_popsize_plot', 'output_likelihood_plot',
        'output_uncertainty', 'adaptive_uncertainty',
        'output_spreadsheets',
        'output_documents', 'output_by_scenario', 'output_horizontally',
        'output_age_calculations', 'riskgroup_diabetes', 'riskgroup_hiv',
        'riskgroup_prison', 'riskgroup_indigenous', 'riskgroup_urbanpoor',
        'riskgroup_ruralpoor',
        'is_lowquality', 'is_amplification', 'is_misassignment', 'is_vary_detection_by_organ',
        'is_timevariant_organs', 'is_timevariant_contactrate', 'is_treatment_history',
        'is_vary_force_infection_by_riskgroup']

    bool_names = {
        'output_uncertainty':
            'Run uncertainty',
        'adaptive_uncertainty':
            'Adaptive search',
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
        'is_timevariant_contactrate':
            'Time-variant contact rate',
        'is_vary_force_infection_by_riskgroup':
            'Heterogeneous mixing',
        'is_treatment_history':
            'Treatment history'}

    for i in range(1, 16):
        bool_keys.append('scenario_' + str(i))

    params = {}

    for key in bool_keys:
        params[key] = {
            'value': False,
            'type': "boolean",
            'label': bool_names[key] if key in bool_names else ''
        }

    defaultBooleanKeys = [
        'riskgroup_diabetes',
        'is_vary_detection_by_organ',
        'is_timevariant_organs',
        'is_treatment_history',
        'adaptive_uncertainty',
        'output_gtb_plots', ]

    for k in defaultBooleanKeys:
        params[k]['value'] = True

    # Model running options
    params['country'] = {
        'type': 'drop_down',
        'options': [
            'Afghanistan', 'Albania', 'Angola', 'Argentina', 'Armenia',
            'Australia', 'Austria', 'Azerbaijan', 'Bahrain', 'Bangladesh',
            'Belarus', 'Belgium', 'Benin', 'Bhutan', 'Botswana', 'Brazil',
            'Bulgaria', 'Burundi', 'Cameroon', 'Chad', 'Chile', 'Croatia',
            'Djibouti', 'Ecuador', 'Estonia', 'Ethiopia', 'Fiji', 'Gabon',
            'Georgia', 'Ghana', 'Guatemala', 'Guinea', 'Philippines', 'Romania'],
        'value': 'Fiji'
    }

    params['integration_method'] = {
        'type': 'drop_down',
        'options': ['Runge Kutta', 'Explicit']
    }
    params['integration_method']['value'] = params['integration_method']['options'][1]

    params['fitting_method'] = {
        'type': 'drop_down',
        'options': ['Method 1', 'Method 2', 'Method 3', 'Method 4', 'Method 5']
    }
    params['fitting_method']['value'] = params['fitting_method']['options'][4]

    params['default_smoothness'] = {
        'type': 'slider',
        'value': 1.0,
        'interval': 0.1,
        'max': 5.0,
    }
    params['time_step'] = {
        'type': 'slider',
        'value': 0.5,
        'max': 0.5,
        'interval': 0.005
    }

    # Model stratifications options
    params['n_organs'] = {
        'type': 'drop_down',
        'options': ['Pos / Neg / Extra', 'Pos / Neg', 'Unstratified']
    }
    params['n_organs']['value'] = params['n_organs']['options'][0]

    params['n_strains'] = {
        'type': 'drop_down',
        'options': ['Single strain', 'DS / MDR', 'DS / MDR / XDR']
    }
    params['n_strains']['value'] = params['n_strains']['options'][0]

    # Uncertainty options
    params['uncertainty_runs'] = {
        'type': 'number',
        'value': 10.0,
        'label': 'Number of uncertainty runs'
    }
    params['burn_in_runs'] = {
        'type': 'number',
        'value': 0,
        'label': 'Number of burn-in runs'
    }
    params['search_width'] = {
        'type': 'number',
        'value': 0.08,
        'label': 'Relative search width'
    }
    params['pickle_uncertainty'] = {
        'type': 'drop_down',
        'options': ['No saving or loading', 'Load', 'Save'],
    }
    params['pickle_uncertainty']['value'] = params['pickle_uncertainty']['options'][0]

    for key, value in params.items():
        if not value.get('label'):
            value['label'] = key

    param_groups = [
        {'keys': [], 'name': 'Model running'},
        {'keys': [], 'name': 'Model Stratifications'},
        {'keys': [], 'name': 'Elaborations'},
        {'keys': [], 'name': 'Scenarios to run'},
        {'keys': [], 'name': 'Uncertainty'},
        {'keys': [], 'name': 'Plotting'},
        {'keys': [], 'name': 'MS Office outputs'}
    ]

    for key in bool_keys:
        name = bool_names[key] if key in bool_names else key
        if ('Plot' in name or 'Draw' in name):
            param_groups[5]['keys'].append(key)
        elif ('uncertainty' in name or 'uncertainty' in key):
            param_groups[4]['keys'].append(key)
        elif 'is_' in key:
            param_groups[2]['keys'].append(key)
        elif ('riskgroup_' in key or 'n_' in key):
            param_groups[1]['keys'].append(key)
        elif 'scenario_' in key:
            param_groups[3]['keys'].append(key)
        else:
            param_groups[6]['keys'].append(key)

    for k in ['country', 'integration_method', 'fitting_method',
              'default_smoothness', 'time_step']:
        param_groups[0]['keys'].append(k)

    for k in ['n_organs', 'n_strains']:
        param_groups[1]['keys'].append(k)

    for k in ['uncertainty_runs', 'burn_in_runs',
              'search_width', 'pickle_uncertainty']:
        param_groups[4]['keys'].append(k)

    return {
        'params': params,
        'paramGroups': param_groups
    }
