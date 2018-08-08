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
import subprocess

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

json_fname = os.path.join(os.path.dirname(__file__), 'country_defaults.json')
if os.path.isfile(json_fname):
    with open(json_fname) as f:
        country_defaults = json.load(f)
else:
    country_defaults = {}


null_attr = {
    "console_lines": [],
    "graph_data": [],
    "is_running": False,
    "project": ''
}


def init():
    save_db_attr(null_attr)

def get_db_attr():
    query = dbmodel.make_obj_query(obj_type="project")
    result = query.all()
    if len(result) > 0:
        obj = result[0]
        attr = obj.attr
        if attr is None:
            attr = copy.deepcopy(null_attr)
            dbmodel.save_object(obj.id, "project", None, null_attr)
    else:
        attr = copy.deepcopy(null_attr)
        dbmodel.create_obj_id(attr=null_attr)
    return attr


def save_db_attr(attr):
    query = dbmodel.make_obj_query(obj_type="project")
    for i, obj in enumerate(query.all()):
        if i == 0:
            dbmodel.save_object(obj.id, "project", None, attr)
        else:
            dbmodel.delete_obj(obj.id)


def public_check_autumn_run():
    return get_db_attr()


def bgui_model_output(output_type, data={}):
    if output_type == "setup":
        save_db_attr({
            "console_lines": [],
            "graph_data": [],
            "is_running": True,
            "project": data['project']
        })
    elif output_type == "init":
        pass
    elif output_type == "console":
        new_lines = data["message"].splitlines()
        attr = get_db_attr()
        attr['is_running'] = True
        for line in new_lines:
            print("> handler.bgui_model_output console: " + line)
        attr['console_lines'].extend(new_lines)
        save_db_attr(attr)
    elif output_type == "graph":
        attr = get_db_attr()
        attr['is_running'] = True
        attr['graph_data'] = copy.deepcopy(data)
        save_db_attr(attr)
    elif output_type == "finish":
        attr = get_db_attr()
        attr["is_running"] = False
        attr["is_completed"] = True
        out_dir = data['out_dir']
        with open(os.path.join(out_dir, 'console.log'), 'w') as f:
            f.write('\n'.join(attr['console_lines']))
        attr.update(get_project_images(out_dir))
        save_db_attr(attr)
    elif output_type == "fail":
        attr = get_db_attr()
        attr["is_running"] = False
        attr["is_completed"] = False
        out_dir = data['out_dir']
        with open(os.path.join(out_dir, 'console.log'), 'w') as f:
            f.write('\n'.join(attr['console_lines']))
        attr.update(get_project_images(out_dir))
        save_db_attr(attr)


def run_model(param_fname):
    print(">> handler.run_model", param_fname)
    with open(param_fname) as f:
        params = json.loads(f.read())
    model_inputs = gui_params.convert_params_to_inputs(params)
    out_dir = os.path.dirname(param_fname)
    autumn_dir = os.path.dirname(autumn.__file__)
    os.chdir(os.path.join(autumn_dir, '..'))
    try:
        model_runner = autumn.model_runner.TbRunner(
            model_inputs, bgui_model_output)
        model_runner.master_runner()
        project = autumn.outputs.Project(
            model_runner, model_inputs, out_dir_project=out_dir)
        project.master_outputs_runner()
        bgui_model_output('finish', {'out_dir': out_dir})
    except Exception as e:
        message = '-------\n'
        message += str(traceback.format_exc())
        message += '-------\n'
        message += 'Error: model crashed'
        bgui_model_output('console', {'message': message})
        bgui_model_output('fail', {'out_dir': out_dir})


def public_run_autumn(params):
    autumn_dir = os.path.join(os.path.dirname(autumn.__file__), os.pardir)
    os.chdir(autumn_dir)

    model_inputs = gui_params.convert_params_to_inputs(params)
    print(">> handler.public_run_autumn", json.dumps(model_inputs, indent=2))

    country = model_inputs['country'].lower()
    save_dir = current_app.config['SAVE_FOLDER']

    project_name = 'test_' + country

    out_dir = os.path.join(save_dir, project_name)
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    param_fname = os.path.abspath(os.path.join(out_dir, 'params.json'))
    with open(param_fname, 'w') as f:
        json.dump(params, f, indent=2)

    bgui_model_output('setup', {'project': project_name})

    this_dir = os.path.dirname(__file__)
    run_local = os.path.join(this_dir, '..', 'run_model.py')
    cmd = ['python', run_local, param_fname]
    print('> public_run_autumn', ' '.join(cmd))

    subprocess.call(cmd)

    return { 'success': True }


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


def get_project_images(out_dir):
    save_dir = os.path.abspath(os.path.dirname(out_dir))

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


def public_get_project_images(project):
    save_dir = current_app.config['SAVE_FOLDER']
    out_dir = os.path.join(save_dir, project)
    return get_project_images(out_dir)


def public_get_example_graph_data():
    import json
    this_dir = os.path.dirname(__file__)
    with open(os.path.join(this_dir, 'graph/graph_data.json')) as f:
        data = json.load(f)
        return {
            'data': data
        }