"""
handler.py
=========

RPC-JSON API Web-handlers

To use define functions with name
- public* public handlers
- login* requires login first
- admin* requires admin login

All handlers must return a JSON dictionary, except for downloadable functions
Handlers can take parameters, which are expected to be only JSON-compatible
Python data structures.

raised Exceptions will be caught and converted into a JSON response
"""

from __future__ import print_function
import os

from flask import session
from flask_login import current_app, current_user, login_user, logout_user

from . import dbmodel


# User handlers

def adminGetUsers():
    return {
        'users': map(dbmodel.parse_user, dbmodel.load_users())
    }


def publicRegisterUser(user_attr):
    username = dbmodel.check_user_attr(user_attr)['username']

    try:
        dbmodel.load_user(username=username)
        raise Exception("User already exists")
    except:

        print("> publicCreateUser user_attr", user_attr)

        created_user_attr = dbmodel.create_user(user_attr)
        return {
            'success': True,
            'user': created_user_attr
        }


def publicGetCurrentUser():
    return dbmodel.parse_user(current_user)


def loginUpdateUser(user_attr):
    return {
        'success': True,
        'user': dbmodel.update_user_from_attr(user_attr)
    }


def publicLoginUser(user_attr):
    if not dbmodel.is_current_user_anonymous():
        print("> publicLoginUser already logged-in")
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

    try:
        user = dbmodel.load_user(**kwargs)
    except:
        raise Exception("User not found")

    print("> publicLoginUser checking hashed password", kwargs, user_attr['password'])
    if user.check_password(user_attr['password']):
        login_user(user)
        return {
            'success': True,
            'user': dbmodel.parse_user(user)
        }

    raise Exception("User/password not found")


def adminDeleteUser(user_id):
    username = dbmodel.delete_user(user_id)['username']
    print("> admin_delete_user ", username)
    return adminGetUsers()


def publicLogoutUser():
    logout_user()
    session.clear()
    return {'success': True}




#############################################################
# PROJECT SPECIFIC HANDLERS
#############################################################

import sys
import json
import glob
import copy
import shutil
import traceback

sys.path.insert(0, os.path.abspath("../.."))
import autumn.model_runner
import autumn.outputs
import autumn.gui_params as gui_params

console_lines = []
is_model_running = False
uncertainty_graph_data = {}

json_fname = os.path.join(os.path.dirname(__file__), 'country_defaults.json')
if os.path.isfile(json_fname):
    with open(json_fname) as f:
        country_defaults = json.load(f)
else:
    country_defaults = {}


def public_check_autumn_run():
    global console_lines
    global is_model_running
    result = {
        "console": console_lines,
        "graph_data": uncertainty_graph_data,
        "is_running": is_model_running
    }
    return result


def bgui_model_output(output_type, data={}):
    if output_type == "init":
        pass
    elif output_type == "console":
        global console_lines
        new_lines = data["message"].splitlines()
        console_lines.extend(new_lines)
        print("> handler.bgui_model_output console: " + '\n'.join(new_lines))
    elif output_type == "graph":
        global uncertainty_graph_data
        print("> handler.bgui_model_output graph")
        uncertainty_graph_data = copy.deepcopy(data)


def public_run_autumn(params):
    global is_model_running
    global console_lines
    global uncertainty_graph_data

    console_lines = []
    uncertainty_graph_data = {}
    is_model_running = True

    autumn_dir = os.path.join(os.path.dirname(autumn.__file__), os.pardir)
    os.chdir(autumn_dir)

    model_inputs = gui_params.convert_params_to_inputs(params)
    print(">> handler.public_run_autumn", json.dumps(model_inputs, indent=2))

    country = model_inputs['country'].lower()
    save_dir = current_app.config['SAVE_FOLDER']
    out_dir = os.path.join(save_dir, 'test_' + country)

    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)

    os.makedirs(out_dir)

    with open(os.path.join(out_dir, 'params.json'), 'w') as f:
        json.dump(params, f, indent=2)

    saved_exception = None

    try:

        model_runner = autumn.model_runner.TbRunner(
            model_inputs, bgui_model_output)
        model_runner.master_runner()

        project = autumn.outputs.Project(
            model_runner, model_inputs, out_dir_project=out_dir)
        project.master_outputs_runner()

        filenames = glob.glob(os.path.join(out_dir, '*png'))
        filenames = [os.path.relpath(p, save_dir) for p in filenames]

        result = {
            'project': os.path.relpath(out_dir, save_dir),
            'success': True,
            'filenames': filenames,
        }

    except Exception as e:
        message = '-------\n'
        message += str(traceback.format_exc())
        message += '-------\n'
        message += 'Error: model crashed'
        bgui_model_output('console', {'message': message})
        result = {
            'project': os.path.relpath(out_dir, save_dir),
            'success': False,
            'filenames': []
        }

    with open(os.path.join(out_dir, 'console.log'), 'w') as f:
        f.write('\n'.join(console_lines))

    is_model_running = False

    if result['success']:
        return result
    else:
        raise Exception('Model crashed')


def public_get_autumn_params():
    result = gui_params.get_autumn_params()
    save_dir = current_app.config['SAVE_FOLDER']
    project_dirs = glob.glob(os.path.join(save_dir, '*'))
    project_dirs = [os.path.relpath(p, save_dir) for p in project_dirs]
    return {
        'params': result['params'],
        'paramGroups': result['param_groups'],
        'projects': project_dirs,
        'countryDefaults': country_defaults
    }

def public_get_project_images(project):
    save_dir = current_app.config['SAVE_FOLDER']
    out_dir = os.path.join(save_dir, project)

    filenames = glob.glob(os.path.join(out_dir, '*png'))
    filenames = [os.path.relpath(p, save_dir) for p in filenames]

    params = None
    json_filename = os.path.join(out_dir, 'params.json')
    if os.path.isfile(json_filename):
        with open(json_filename) as f:
            params = json.load(f)

    console_lines = []
    console_filename = os.path.join(out_dir, 'console.log')
    if os.path.isfile(console_filename):
        with open(console_filename) as f:
            text = f.read()
            console_lines = text.splitlines()

    return {
        'filenames': filenames,
        'params': params,
        'consoleLines': console_lines
    }

def public_get_example_graph_data():
    import json
    this_dir = os.path.dirname(__file__)
    with open(os.path.join(this_dir, 'graph/graph_data.json')) as f:
        data = json.load(f)
        return {
            'data': data
        }