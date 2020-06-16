"""
Build the COVID country model runner functions
"""
import os
import yaml

from autumn.model_runner import build_model_runner
from autumn.tool_kit.params import load_params

from .plots import load_plot_config
from .model import build_model


class Country:
    AUSTRALIA = "australia"
    PHILIPPINES = "philippines"
    MALAYSIA = "malaysia"
    VICTORIA = "victoria"
    LIBERIA = "liberia"
    MANILA = "manila"
    CALABARZON = "calabarzon"
    BICOL = "bicol"
    CENTRALVISAYAS = "centralVisayas"
    COUNTRIES = [AUSTRALIA, PHILIPPINES, MALAYSIA, VICTORIA, LIBERIA, MANILA, CALABARZON, BICOL, CENTRALVISAYAS]


FILE_DIR = os.path.dirname(os.path.abspath(__file__))
POST_PROCESSING_PATH = os.path.join(FILE_DIR, "post-processing.yml")

with open(POST_PROCESSING_PATH, "r") as f:
    pp_config = yaml.safe_load(f)


class CountryModel:
    def __init__(self, country: str):
        self.country = country
        self._run_model = build_model_runner(
            model_name=f"covid_{country}",
            build_model=self.build_model,
            params=self.params,
            post_processing_config=pp_config,
            plots_config=self.plots_config,
        )

    def build_model(self, params):
        return build_model(params)

    @property
    def params(self):
        return load_params("covid_19", self.country)

    @property
    def plots_config(self):
        return load_plot_config(self.country)

    def run_model(self, run_name="model-run", run_desc=""):
        self._run_model(run_name, run_desc)
