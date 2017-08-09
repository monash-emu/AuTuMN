from flask.ext.sqlalchemy import SQLAlchemy

# This is a module-wide location to connect to the database initialized by api.py
db = None

def connect_to_app(app):
    global db
    db = SQLAlchemy(app)

