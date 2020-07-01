"""
Build the Marshall Islands model runner
"""
import os
import yaml

from autumn.model_runner import build_model_runner

from . import calibration
from .model import build_model

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PARAMS_PATH = os.path.join(FILE_DIR, "params.yml")
POST_PROCESSING_PATH = os.path.join(FILE_DIR, "post-processing.yml")
PLOTS_PATH = os.path.join(FILE_DIR, "plots.yml")

with open(PARAMS_PATH, "r") as f:
    params = yaml.safe_load(f)


run_model = build_model_runner(
    model_name="marshall_islands", param_set_name=None, build_model=build_model, params=params,
)
