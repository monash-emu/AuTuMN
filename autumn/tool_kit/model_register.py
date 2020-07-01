"""
A tool for application models to register themselves so that they serve a standard interface
"""
from abc import ABC, abstractmethod

from autumn.model_runner import build_model_runner
from autumn.constants import Region


class RegionAppBase(ABC):
    @abstractmethod
    def build_model(self, params):
        pass

    @abstractmethod
    def run_model(self, *args, **kwargs):
        pass

    @property
    @abstractmethod
    def params(self):
        pass

    @property
    @abstractmethod
    def plots_config(self):
        pass


class App:
    def __init__(self):
        self.region_names = []
        self.region_apps = {}

    def register(self, region_app: RegionAppBase):
        name = region_app.region
        assert name not in self.region_names
        self.region_names.append(name)
        self.region_apps[name] = region_app

    def get_region_app(self, region: str):
        return self.region_apps[region]
