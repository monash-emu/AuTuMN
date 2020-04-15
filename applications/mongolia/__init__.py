"""
Build the Mongolia model runner
"""
import os
import yaml

from autumn.model_runner import build_model_runner
from autumn.tool_kit.utils import merge_dicts

from .model import build_model

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PARAMS_PATH = os.path.join(FILE_DIR, "params.yml")
MLE_PARAMS_PATH = os.path.join(FILE_DIR, "params-mle.yml")
POST_PROCESSING_PATH = os.path.join(FILE_DIR, "post-processing.yml")
PLOTS_PATH = os.path.join(FILE_DIR, "plots.yml")


with open(PARAMS_PATH, "r") as f:
    params = yaml.safe_load(f)

with open(MLE_PARAMS_PATH, "r") as f:
    mle_params = yaml.safe_load(f)

# Incorporate max likelihood params from MCMC runs
params["default"] = merge_dicts(mle_params, params["default"])

with open(POST_PROCESSING_PATH, "r") as f:
    pp_config = yaml.safe_load(f)

with open(PLOTS_PATH, "r") as f:
    plots_config = yaml.safe_load(f)

run_model = build_model_runner(
    model_name="mongolia",
    build_model=build_model,
    params=params,
    post_processing_config=pp_config,
    plots_config=plots_config,
)
