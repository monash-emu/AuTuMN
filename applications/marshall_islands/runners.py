"""
Build the Marshall Islands model runner
"""
import os
import yaml

from autumn.model_runner import build_model_runner

from .rmi_model import build_rmi_model

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PARAMS_PATH = os.path.join(FILE_DIR, "params.yml")
OUTPUTS_PATH = os.path.join(FILE_DIR, "outputs.yml")


with open(PARAMS_PATH, "r") as f:
    params = yaml.safe_load(f)

with open(OUTPUTS_PATH, "r") as f:
    outputs = yaml.safe_load(f)

run_rmi_model = build_model_runner(
    model_name="marshall_islands", build_model=build_rmi_model, params=params, outputs=outputs,
)
