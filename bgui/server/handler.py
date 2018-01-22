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
import traceback

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
import autumn.gui_params as gui_params


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
    model_inputs = gui_params.convert_params_to_inputs(params)
    print(">> handler.public_run_autumn", json.dumps(model_inputs, indent=2))
    bgui_output_lines = []
    autumn_dir = os.path.join(os.path.dirname(autumn.__file__), os.pardir)
    os.chdir(autumn_dir)
    try:
        model_runner = autumn.model_runner.ModelRunner(
            model_inputs, None, bgui_model_output)
        model_runner.master_runner()
        project = autumn.outputs.Project(model_runner, model_inputs)
        project.master_outputs_runner()
        result = {'success': True}
    except Exception:
        result = {'success': False}
        traceback.print_exc()
    is_bgui_running = False
    return result


def public_get_autumn_params():
    result = gui_params.get_autumn_params()
    return {
        'params': result['params'],
        'paramGroups': result['param_groups']
    }
