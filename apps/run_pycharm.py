"""
Entry point for PyCharm users to run an application
"""

from apps import covid_19, marshall_islands, mongolia


RUN_NAME = "manual-calibration"
RUN_DESCRIPTION = "trying to x and y the z"

# marshall_islands.run_model()
# covid_19.phl.run_model(RUN_NAME)
# covid_19.mys.run_model(RUN_NAME)
# covid_19.aus.run_model(RUN_NAME, RUN_DESCRIPTION)
covid_19.vic.run_model(RUN_NAME, RUN_DESCRIPTION)
