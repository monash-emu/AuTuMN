"""
Entry point for PyCharm users to run an application
"""

from apps import covid_19, marshall_islands, mongolia, sir_example

# from autumn.plots.database_plots import plot_from_database

RUN_NAME = "manual-calibration"
RUN_DESCRIPTION = "trying to x and y the z"

# marshall_islands.run_model()
# covid_19.phl.run_model(RUN_NAME)
# covid_19.mys.run_model(RUN_NAME)
# covid_19.lbr.run_model(RUN_NAME)
# covid_19.aus.run_model(RUN_NAME, RUN_DESCRIPTION)
covid_19.vic.run_model(RUN_NAME, RUN_DESCRIPTION)

# Simple SIR model for demonstration
# sir_example.aus.run_model(RUN_NAME, RUN_DESCRIPTION)
# sir_example.phl.run_model(RUN_NAME, RUN_DESCRIPTION)

# MODEL_RUN_PATH = "data/covid_victoria/model-run-27-04-2020--17-06-42/"
# plot_from_database(MODEL_RUN_PATH)
