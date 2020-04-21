"""
Build the COVID country model runner functions
"""
import os
import yaml

from autumn.model_runner import build_model_runner
from autumn.tool_kit.params import load_params

from .covid_model import AUSTRALIA, PHILIPPINES, MALAYSIA, VICTORIA, build_covid_model
from .covid_matrices import build_covid_matrices

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_PATH = os.path.join(FILE_DIR, "outputs.yml")

with open(OUTPUTS_PATH, "r") as f:
    outputs = yaml.safe_load(f)


def build_country_runner(country: str):
    """
    Factory function to build country specific model runner.
    """

    def build_country_covid_model(update_params={}):
        """
        Build country-specific COVID model
        """
        return build_covid_model(country, update_params)

    params = load_params(FILE_DIR, application=country)
    country_name = country.lower()
    return build_model_runner(
        model_name=f"covid_{country_name}",
        build_model=build_country_covid_model,
        params=params,
        outputs=outputs
        # mixing_functions=build_covid_matrices(params["default"]["country"], params["mixing"]),
    )


run_covid_aus_model = build_country_runner(AUSTRALIA)
run_covid_phl_model = build_country_runner(PHILIPPINES)
run_covid_mys_model = build_country_runner(MALAYSIA)
run_covid_vic_model = build_country_runner(VICTORIA)
