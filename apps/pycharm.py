"""
Entry point for PyCharm users to run an application
"""
from autumn.constants import Region
from apps import covid_19, sir_example

# Run a COVID model manually.
app_region = covid_19.app.get_region(Region.DHHS)
app_region.run_model(run_scenarios=True)

# Simple SIR model for demonstration
#app_region = sir_example.app.get_region(Region.AUSTRALIA)
#app_region.run_model()

# # Run a calibration
# app_region = covid_19.app.get_region(Region.VICTORIA)
# app_region.calibrate_model(max_seconds=30, run_id=1, num_chains=1)
