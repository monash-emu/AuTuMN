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
"""

from __future__ import print_function
import os
import time

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

    print("> publicLoginUser loading", kwargs, user_attr['password'])
    user = dbmodel.load_user(**kwargs)

    if user.check_password(user_attr['password']):
        login_user(user)
        return {
            'success': True,
            'user': dbmodel.parse_user(user)
        }

    raise Exception("User not found")


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
import traceback
import glob

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
        bgui_output_lines.extend(data["message"].splitlines())
        print(">> handler.bgui_model_output console:", data["message"])
    elif output_type == "uncertainty_graph":
        pass


def public_run_autumn(params):
    """
    Run the model
    """
    global is_bgui_running
    global bgui_output_lines
    is_bgui_running = True
    bgui_output_lines = []

    autumn_dir = os.path.join(os.path.dirname(autumn.__file__), os.pardir)
    os.chdir(autumn_dir)

    model_inputs = gui_params.convert_params_to_inputs(params)
    print(">> handler.public_run_autumn", json.dumps(model_inputs, indent=2))

    try:
        model_runner = autumn.model_runner.TbRunner(
            model_inputs, bgui_model_output)
        model_runner.master_runner()

        project = autumn.outputs.Project(model_runner, model_inputs)
        out_dir = project.master_outputs_runner()

        # out_dir = "/Users/boscoh/Projects/AuTuMN/projects/test_fiji"

        save_dir = current_app.config['SAVE_FOLDER']
        filenames = glob.glob(os.path.join(out_dir, '*'))
        filenames = [os.path.relpath(p, save_dir) for p in filenames]

        result = {
            'project': os.path.relpath(out_dir, save_dir),
            'success': True,
            'filenames': filenames,
        }

    except Exception:
        result = {'success': False}
        traceback.print_exc()

    is_bgui_running = False

    return result


def public_get_autumn_params():
    result = gui_params.get_autumn_params()
    save_dir = current_app.config['SAVE_FOLDER']
    project_dirs = glob.glob(os.path.join(save_dir, '*'))
    project_dirs = [os.path.relpath(p, save_dir) for p in project_dirs]
    return {
        'params': result['params'],
        'paramGroups': result['param_groups'],
        'projects': project_dirs
    }

def public_get_project_images(project):
    save_dir = current_app.config['SAVE_FOLDER']
    out_dir = os.path.join(save_dir, project)
    filenames = glob.glob(os.path.join(out_dir, '*'))
    filenames = [os.path.relpath(p, save_dir) for p in filenames]

    return {
        'success': True,
        'filenames': filenames,
    }
