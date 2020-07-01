"""
Entry point for PyCharm users to run an application
"""
from autumn.constants import Region
from autumn.plots.database_plots import plot_from_database
from apps import covid_19, marshall_islands, mongolia, sir_example
from apps.covid_19.calibration import get_calibration_func

## Run a COVID model manually.
REGION = Region.MALAYSIA
RUN_SCENARIOS = True
region_app = covid_19.get_region_app(REGION)
region_app.run_model(RUN_SCENARIOS)

## Simple SIR model for demonstration
# REGION = Region.AUSTRALIA
# region_app = sir_example.get_region_app(REGION)
# region_app.run_model()

## Plot an existing model's run data to files.
# MODEL_RUN_PATH = "data/covid_victoria/model-run-27-04-2020--17-06-42/"
# plot_from_database(MODEL_RUN_PATH)


## Run a calibration
# MAX_SECONDS = 120
# CHAIN_ID = 0
# calibrate_func = covid_19.calibration.get_calibration_func(REGION)
# calibrate_func(MAX_SECONDS, CHAIN_ID)
