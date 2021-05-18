"""
Entry point for PyCharm users to run an application
"""
from autumn.settings import Region, Models
from autumn.tools.project import get_project

region = Region.FRANCE
model = Models.COVID_19

project = get_project(model, region)

# Run a COVID model manually.
project.run_model(run_scenarios=False)

# Run a calibration
# project.calibrate_model(max_seconds=60, run_id=1, num_chains=1)
