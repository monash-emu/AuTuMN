"""
A tool for application models to register themselves so that they serve a standard interface
"""
from abc import ABC, abstractmethod

from autumn.model_runner import build_model_runner
from autumn.constants import Region
from autumn.tool_kit.params import load_params, load_targets


class AppRegion:
    def __init__(self, app_name: str, region_name: str, build_model, calibrate_model):
        self.region_name = region_name
        self.app_name = app_name
        self._build_model = build_model
        self._calibrate_model = calibrate_model

    @property
    def params(self):
        return load_params(self.app_name, self.region_name)

    @property
    def targets(self):
        return load_targets(self.app_name, self.region_name)

    def calibrate_model(self, max_seconds: int, run_id: int, num_chains: int):
        return self._calibrate_model(max_seconds, run_id, num_chains)

    def build_model(self, params: dict):
        return self._build_model(params)

    def run_model(self, *args, **kwargs):
        run_model = build_model_runner(
            model_name=self.app_name,
            param_set_name=self.region_name,
            build_model=self._build_model,
            params=self.params,
        )
        return run_model(*args, **kwargs)


class App:
    def __init__(self, app_name):
        self.app_name = app_name
        self.region_names = []
        self.region_modules = {}

    def register(self, app_region):
        assert app_region.region_name not in self.region_names
        assert app_region.app_name == self.app_name
        self.region_names.append(app_region.region_name)
        self.region_modules[app_region.region_name] = app_region

    def get_region(self, region_name: str) -> AppRegion:
        return self.region_modules[region_name]
