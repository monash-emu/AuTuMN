import os
from pathlib import Path
from os.path import dirname

# Filesystem paths
BASE_PATH = Path(os.path.abspath(dirname(dirname(dirname(__file__)))))
DATA_PATH = BASE_PATH / "data"
DOCS_PATH = os.path.join(BASE_PATH, "docs")
MODELS_PATH = os.path.join(BASE_PATH, "autumn", "models")
PROJECTS_PATH = os.path.join(BASE_PATH, "autumn", "projects")
INPUT_DATA_PATH = os.path.join(DATA_PATH, "inputs")
OUTPUT_DATA_PATH = DATA_PATH / "outputs"
REMOTE_PATH = OUTPUT_DATA_PATH / "remote"
REMOTE_BASE_DIR = OUTPUT_DATA_PATH / "remote"
LOGGING_DIR = BASE_PATH / "log"
