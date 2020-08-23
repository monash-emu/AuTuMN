"""
Entry point for PyCharm users to run an application
"""
from autumn.constants import Region
from apps import covid_19, sir_example, tuberculosis

# Run a COVID model manually.
app_region = covid_19.app.get_region(Region.BARWON_SOUTH_WEST)
app_region.run_model(run_scenarios=True)

# Simple SIR model for demonstration
#app_region = sir_example.app.get_region(Region.AUSTRALIA)
#app_region.run_model()


# app_region = tuberculosis.app.get_region(Region.PHILIPPINES)
# app_region.run_model(run_scenarios=True)
# app_region.calibrate_model(max_seconds=20, run_id=0, num_chains=1)


# # Run a calibration
# app_region = covid_19.app.get_region(Region.BARWON_SOUTH_WEST)
# app_region.calibrate_model(max_seconds=30, run_id=1, num_chains=1)
