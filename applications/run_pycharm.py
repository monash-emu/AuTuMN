"""
Entry point for PyCharm users to run an application
"""

# from applications.marshall_islands.runners import run_rmi_model
from applications.covid_19.runners import \
    run_covid_aus_model, run_covid_phl_model, run_covid_vic_model, run_covid_mys_model

RUN_NAME = "manual-calibration"
RUN_DESCRIPTION = "trying to x and y the z"

# run_rmi_model()
# run_covid_phl_model(RUN_NAME)
run_covid_mys_model(RUN_NAME)
# run_covid_aus_model(RUN_NAME, RUN_DESCRIPTION)
# run_covid_vic_model(RUN_NAME, RUN_DESCRIPTION)
