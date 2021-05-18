import os
from os.path import dirname

# Filesystem paths
BASE_PATH = os.path.abspath(dirname(dirname(dirname(__file__))))
DATA_PATH = os.path.join(BASE_PATH, "data")
PROJECTS_PATH = os.path.join(BASE_PATH, "projects")
INPUT_DATA_PATH = os.path.join(DATA_PATH, "inputs")
OUTPUT_DATA_PATH = os.path.join(DATA_PATH, "outputs")
REMOTE_PATH = os.path.join(OUTPUT_DATA_PATH, "remote")
REMOTE_BASE_DIR = os.path.join(OUTPUT_DATA_PATH, "remote")
LOGGING_DIR = os.path.join(BASE_PATH, "log")
