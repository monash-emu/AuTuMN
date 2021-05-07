"""
Entry point for PyCharm users to run an application
"""
import os

from apps import covid_19, sir_example, tuberculosis, tuberculosis_strains
from autumn.region import Region

from apps.covid_19.mixing_optimisation.constants import OPTI_REGIONS
os.chdir("..")  # Make repo root the current directory

# Run a COVID model manually.
# app_region = covid_19.app.get_region(Region.FRANCE)
# app_region.run_model(run_scenarios=False)

# Simple SIR model for demonstration
# app_region = sir_example.app.get_region(Region.AUSTRALIA)
# app_region.run_model()M


# app_region = tuberculosis.app.get_region(Region.LODDON_MALLEE)
# app_region.run_model(run_scenarios=True)
# app_region.calibrate_model(max_seconds=20, run_id=0, num_chains=1)


# # Run a calibration
# app_region = covid_19.app.get_region(Region.VICTORIA)
# app_region.calibrate_model(max_seconds=60, run_id=1, num_chains=1)

 # Run a calibration
app_region = tuberculosis_strains.app.get_region(Region.PHILIPPINES)
app_region.calibrate_model(max_seconds=60, run_id=1, num_chains=1)

# Used by Romain, please do not delete
# for region in OPTI_REGIONS:
#     app_region = covid_19.app.get_region(region)
#     app_region.calibrate_model(max_seconds=5, run_id=1, num_chains=1)
