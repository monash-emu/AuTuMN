"""

handler.py
=========

Contains all the functions that fetches and saves optima objects to/from database
and the file system. These functions abstracts out the data i/o for the web-server
api calls.

Function call pairs are load_*/save_* and refers to saving to database.
(create, update, load, delete)

Database record variables should have suffix _record

Parsed data structures should have suffix _summary

All parameters and return types are either id's, json-summaries, or mpld3 graphs

"""

from __future__ import print_function
import os
import sys

from flask import current_app, session, abort
from flask.ext.login import current_user, login_user, logout_user
from validate_email import validate_email

# User handlers

import dbmodel


def check_valid_email(email):
    if not email:
        return email
    if validate_email(email):
        return email
    raise ValueError('{} is not a valid email'.format(email))


def check_sha224_hash(password):
    if isinstance(password, basestring) and len(password) == 56:
        return password
    raise ValueError('Invalid password - expecting SHA224')


def check_user_attr(user_attr):
    return {
        'email': check_valid_email(user_attr.get('email', None)),
        'name': user_attr.get('name', ''),
        'username': user_attr.get('username', ''),
        'password': check_sha224_hash(user_attr.get('password')),
    }


def is_anonymous():
    try:
        userisanonymous = current_user.is_anonymous()
    except:
        userisanonymous = current_user.is_anonymous
    return userisanonymous


def admin_retrieve_users():
    return {'users': map(dbmodel.parse_user, dbmodel.load_users())}


def public_create_user(user_attr):
    username = check_user_attr(user_attr)['username']
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
    if not is_anonymous():
        print(">> public_login_user already logged-in")
        return {
            'success': True,
            'user': dbmodel.parse_user(current_user)
        }

    user_attr = check_user_attr(user_attr)
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
        print(">> public_login_user compare", user.password, user_attr['password'])
        if user.password == user_attr['password']:
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


## DIRECTORIES

def get_user_server_dir(dirpath, user_id=None):
    """
    Returns a user directory if user_id is defined
    """
    try:
        if not is_anonymous():
            current_user_id = user_id if user_id else current_user.id
            user_path = os.path.join(dirpath, str(current_user_id))
            if not (os.path.exists(user_path)):
                os.makedirs(user_path)
            return user_path
    except:
        return dirpath
    return dirpath


def get_server_filename(filename):
    """
    Returns the path to save a file on the server
    """
    dirname = get_user_server_dir(current_app.config['SAVE_FOLDER'])
    if not (os.path.exists(dirname)):
        os.makedirs(dirname)
    if os.path.dirname(filename) == '' and not os.path.exists(filename):
        filename = os.path.join(dirname, filename)
    return filename


# RPC-JSON API Web-handlers
#
# To use define functions with name
# - public_* public handlers
# - login_* requires login frist
# - admin_* requires admin login

bgui_output_lines = []
is_bgui_running = False

sys.path.insert(0, os.path.abspath("../.."))
import autumn.model_runner
import autumn.outputs


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
        print(">> handler.bgui_model_output uncertainty_graph:", data)


def public_run_autumn(gui_outputs):
    """
    Run the model
    """
    global is_bgui_running
    global bgui_output_lines
    is_bgui_running = True
    bgui_output_lines = []
    autumn_dir = os.path.join(os.path.dirname(autumn.__file__), os.pardir)
    os.chdir(autumn_dir)
    try:
        model_runner = autumn.model_runner.ModelRunner(
            gui_outputs, None, None, bgui_model_output)
        model_runner.master_runner()
        project = autumn.outputs.Project(model_runner, gui_outputs)
        project.master_outputs_runner()
        result = {'success': True}
    except:
        result = {'success': False}
    is_bgui_running = False
    return result
