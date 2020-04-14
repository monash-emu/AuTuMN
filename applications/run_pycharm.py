"""
Entry point for PyCharm users to run an application
"""

from marshall_islands.runnners import run_rmi_model
from covid_19.runnners import run_covid_aus_model, run_covid_phl_model

RUN_NAME = "manual-calibration"
RUN_DESCRIPTION = "trying to x and y the z"

# run_rmi_model()
# run_covid_phl_model()
run_covid_aus_model(RUN_NAME, RUN_DESCRIPTION)
