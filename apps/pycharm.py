"""
Entry point for PyCharm users to run an application
"""
import os

from autumn.constants import Region
from apps import covid_19, sir_example, tuberculosis

os.chdir('..')  # Make repo root the current directory

# Run a COVID model manually.
app_region = covid_19.app.get_region(Region.CENTRAL_VISAYAS)
app_region.run_model(run_scenarios=False)

# Simple SIR model for demonstration
# app_region = sir_example.app.get_region(Region.AUSTRALIA)
# app_region.run_model()M


# app_region = tuberculosis.app.get_region(Region.PHILIPPINES)
# app_region.run_model(run_scenarios=True)
# app_region.calibrate_model(max_seconds=20, run_id=0, num_chains=1)


# # Run a calibration
# app_region = covid_19.app.get_region(Region.BARWON_SOUTH_WEST)
# app_region.calibrate_model(max_seconds=30, run_id=1, num_chains=1)
