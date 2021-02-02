import os

# Filesystem paths
file_path = os.path.abspath(__file__)
separator = "\\" if "\\" in file_path else "/"
BASE_PATH = separator.join(file_path.split(separator)[:-2])
DATA_PATH = os.path.join(BASE_PATH, "data")
INPUT_DATA_PATH = os.path.join(DATA_PATH, "inputs")
OUTPUT_DATA_PATH = os.path.join(DATA_PATH, "outputs")
REMOTE_PATH = os.path.join(OUTPUT_DATA_PATH, "remote")
APPS_PATH = os.path.join(BASE_PATH, "apps")


REMOTE_BASE_DIR = os.path.join(OUTPUT_DATA_PATH, "remote")
