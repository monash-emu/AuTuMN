import os
import yaml

from autumn.model_runner import build_model_runner
from autumn.tool_kit.params import load_params
from autumn.tool_kit.model_register import RegionAppBase

from .model import build_model


class RegionApp(RegionAppBase):
    def __init__(self, region: str):
        self.region = region
        self._run_model = build_model_runner(
            model_name="sir_example",
            param_set_name=self.region,
            build_model=self.build_model,
            params=self.params,
        )

    def build_model(self, params):
        return build_model(params)

    def run_model(self, *args, **kwargs):
        self._run_model(*args, **kwargs)

    @property
    def params(self):
        return load_params("sir_example", self.region)

    @property
    def plots_config(self):
        return None
