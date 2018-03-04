"""
Major global app variables defined here so that
other modules can easily access them
"""

from flask import Flask
from flask_sqlalchemy import SQLAlchemy

# Create flask app
app = Flask(__name__)

# Load configuration
app.config.from_pyfile('config.py')

# Connect to database as defined in config.py
db = SQLAlchemy(app)


