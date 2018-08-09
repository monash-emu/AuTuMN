'''
Defines the flask app - URL responses for a WSGI gateway.

This file is designed to be run from `run_local.py`, in the parent directory
To test with only the flask app:

    export FLASK_APP=api.py
    flask run

'''

from __future__ import print_function
import os
import logging
from functools import wraps
import traceback
import json

from flask import abort, jsonify, current_app, request, helpers, \
    json, make_response, send_from_directory, send_file
from flask_login import LoginManager, current_user
from werkzeug.utils import secure_filename

from . import conn
from . import dbmodel
from . import handler

# Load app from singleton global module
app = conn.app

app.logger.setLevel(logging.DEBUG)

handler.init()

# Setup login manager with dbmodel.UserDb
login_manager = LoginManager()
login_manager.init_app(app)


@login_manager.user_loader
def load_user(user_id):
    try:
        user = dbmodel.load_user(id=user_id)
    except Exception:
        user = None
    return user


@login_manager.unauthorized_handler
def unauthorized_handler():
    abort(401)


# Setup RPC handlers

def report_exception_decorator(api_call):
    @wraps(api_call)
    def _report_exception(*args, **kwargs):
        from werkzeug.exceptions import HTTPException
        try:
            return api_call(*args, **kwargs)
        except Exception as e:
            exception = traceback.format_exc()
            # limiting the exception information to 10000 characters maximum
            # (to prevent monstrous sqlalchemy outputs)
            current_app.logger.error('Exception during request %s: %.10000s' % (request, exception))
            if isinstance(e, HTTPException):
                raise
            code = 500
            reply = {'exception': exception}
            return make_response(jsonify(reply), code)

    return _report_exception


def get_post_data_json():
    return json.loads(request.data)


def run_method(method, params):
    if method.startswith('admin'):
        if (current_user.is_anonymous()) \
                or not current_user.is_authenticated() \
                or not current_user.is_admin:
            return app.login_manager.unauthorized()
    elif method.startswith('login'):
        if not current_user.is_authenticated():
            return app.login_manager.unauthorized()
    elif not method.startswith('public'):
        raise ValueError('Function "%s" not valid' % (method))

    if hasattr(handler, method):
        fn = getattr(handler, method)
        app.logger.info('run_method %s %s' % (method, params))
    else:
        app.logger.info('run_method: error: function "%s" does not exist' % (method))
        raise ValueError('Function "%s" does not exist' % (method))

    return fn(*params)


# NOTE: twisted wgsi only serves url's with /api/*


@app.route('/api', methods=['GET'])
def root():
    return json.dumps({"rpcJsonVersion": "2.0"})


@app.route('/api/rpc-run', methods=['POST'])
@report_exception_decorator
def run_remote_procedure():
    """
    post-data:
        'method': string name of function in handler
        'params': list of arguments for the function
    """
    json = get_post_data_json()
    method = json['method']
    params = json.get('params', [])

    try:
        result = run_method(method, params)
        return jsonify({
            "result": result,
            "jsonrpc": "2.0"
        })
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({
            "error": {
                "code": -1,
                "message": str(e)
            },
            "jsonrpc": "2.0"
        })


@app.route('/api/rpc-download', methods=['POST'])
@report_exception_decorator
def send_downloadable_file():
    """
    post-body-json:
        'method': string name of function in handler
        'params': list of arguments for the function
    """
    post_data = get_post_data_json()
    method = post_data['method']
    params = post_data.get('params', [])

    try:
        result = run_method(method, params)
        filename = result['filename']
        dirname = os.path.dirname(filename)
        basename = os.path.basename(filename)

        response = helpers.send_from_directory(
            dirname,
            basename,
            as_attachment=True,
            attachment_filename=basename)

        response.status_code = 201

        response.headers['data'] = json.dumps({
            'result': result['data'],
            "jsonrpc": "2.0"
        })

        response.headers['filename'] = basename
        response.headers['Access-Control-Expose-Headers'] = 'data, filename'

        return response

    except Exception as e:
        print(traceback.format_exc())

        response = make_response()
        response.headers['data'] = json.dumps({
            "error": {
                "code": -1,
                "message": str(e)
            },
            "jsonrpc": "2.0"
        })
        response.headers['filename'] = ''
        response.headers['Access-Control-Expose-Headers'] = 'data, filename'

        return response


@app.route('/api/rpc-upload', methods=['POST'])
@report_exception_decorator
def receive_uploaded_file():
    """
    file-upload
    request-form:
        method: name of function
        params: string of JSON.stringify object
    """
    files = request.files.getlist('uploadFiles')

    dirname = current_app.config['SAVE_FOLDER']
    if not (os.path.exists(dirname)):
        os.makedirs(dirname)

    server_filenames = []
    for file in files:
        basename = secure_filename(file.filename)
        filename = os.path.join(dirname, basename)
        file.save(filename)
        server_filenames.append(filename)

    method = request.form.get('method')
    params = json.loads(request.form.get('params', '[]'))
    params.insert(0, server_filenames)

    try:
        result = run_method(method, params)
        return jsonify({
            "result": result,
            "jsonrpc": "2.0"
        })
    except Exception as e:
        print(traceback.format_exc())

        return jsonify({
            "error": {
                "code": -1,
                "message": str(e)
            },
            "jsonrpc": "2.0"
        })


# use this to set absolute paths on init
this_dir = os.path.join(os.getcwd(), os.path.dirname(__file__))


# Set SAVE_FOLDER to absolute path on initialization as directories
# can get scrambled later on
app.config['SAVE_FOLDER'] = os.path.abspath(
    os.path.join(this_dir, app.config['SAVE_FOLDER']))
app.logger.info('SAVE_FOLDER: ' + app.config['SAVE_FOLDER'])


# Route to load files saved on the server from uploads
@app.route('/file/<path:path>', methods=['GET'])
def serve_saved_file(path):
    app.logger.info("FILE: " + os.path.join(app.config['SAVE_FOLDER'], path))
    return send_from_directory(app.config['SAVE_FOLDER'], path)


# These routes are to load in the compiled web-client from the
# same IP:PORT as the server
app.static_folder = os.path.join(this_dir, app.config['STATIC_FOLDER'])
app.logger.info('static_folder: ' + app.static_folder)


@app.route('/')
def index():
    return send_file(os.path.join(app.static_folder, '../index.html'))


# http://reputablejournal.com/adventures-with-flask-cors.html#.WW6-INOGMm8
# Allow Cross-Origin-Resource-Sharing, mainly for working with hot reloading webclient
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:8080')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response


