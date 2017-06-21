from flask import Flask, render_template
from flask_socketio import SocketIO, emit

from control import Autumn

import eventlet
eventlet.monkey_patch()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'AutumnLeaves'
socketio = SocketIO(app, async_mode='eventlet', ping_timeout=6000000, async_handlers=True)

# Ping-Timeout is important, since the server otherwise pings the cient regularly and disrupts connections in
# long-running functions, e.g. uncertainty. Set to 6000 seconds, which shoudl be enough. Might need to think
# about using a synchronous interface with AJAX calls, rather than asynchronous interface with Websockets.
# Note that we are monkey-patching several base functions of Python to be compatible with Flask-SocketIO.
# I think this is necessary to empty the message queue for emission, i.e. calls to emit(), at the beginning
# of each loop iteration in which the occur - otherwise messages get delayed until loops are finished and send
# as a single batch to the GUI.

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('run_model', namespace='/autumn')
def run_model(data):

    """

    On pressing the Start button in the GUI, print the starting message (Python) and
    initiate the Autumn class, which is the control module for converting the settings into
    an appropriate format and initiate the ModelRunner and Output modules - future
    implementations will need a stopping functionality (see below).

    The function receives data sent from JS, which contains the starting message ("message")
    and all settings specified in the GUI ("data").

    """

    message = data["message"]
    settings = data["data"]

    print(message)

    autumn = Autumn(gui_settings=settings)


@socketio.on('connect', namespace='/autumn')
def test_connect():

    """
    On connection to the client (JS) print a message (Python) and emit a response, which will
    be logged in the console of the browser (JS).

    """

    print("Client (JS) is connected to AuTuMN.")
    emit('connection_response', {'message': 'Server (Python) AuTuMN is now connected.'})

@socketio.on('disconnect', namespace='/autumn')
def test_disconnect():
    """
    On disconnection to the client (JS) print a message (Python) and emit a response, which will
    be logged in the console of the browser (JS).

    """

    print('Client disconnected from AuTuMN')
    emit('connection_response', {'message': 'Server (Python) AuTuMN is now disconnected.'})

@socketio.on('stop_model', namespace="/autumn")
def stop_model():
    """

    On pressing the Stop button in the GUI, print the stopping message (Python) and emit
    a response, which will be logged in the console of the GUI (JS).

    This function will be extended to stop the model run...

    """

    print("Stopping model...")
    emit('console', {'message': "Stopping model..."})


if __name__ == '__main__':
    socketio.run(app, debug=True)

