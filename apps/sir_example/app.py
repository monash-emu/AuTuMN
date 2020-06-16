import os
import yaml

from autumn.model_runner import build_model_runner
from autumn.tool_kit.params import load_params
from autumn.tool_kit.model_register import RegionAppBase

from .model import build_model


FILE_DIR = os.path.dirname(os.path.abspath(__file__))
POST_PROCESSING_PATH = os.path.join(FILE_DIR, "post-processing.yml")
PLOTS_PATH = os.path.join(FILE_DIR, "plots.yml")

with open(POST_PROCESSING_PATH, "r") as f:
    pp_config = yaml.safe_load(f)

with open(PLOTS_PATH, "r") as f:
    plots_config = yaml.safe_load(f)


class RegionApp(RegionAppBase):
    def __init__(self, region: str):
        self.region = region
        self._run_model = build_model_runner(
            model_name=f"sir_example_{region}",
            build_model=self.build_model,
            params=self.params,
            post_processing_config=pp_config,
            plots_config=plots_config,
        )

    def build_model(self, params):
        return build_model(params)

    def run_model(self, run_name="model-run", run_desc=""):
        self._run_model(run_name, run_desc)

    @property
    def params(self):
        return load_params("sir_example", self.region)

    @property
    def plots_config(self):
        return None
