"""
Entry point for PyCharm users to run an application
"""
from autumn.settings import Region, Models
from autumn.tools.project import get_project, run_project_locally

region = Region.VICTORIA
model = Models.COVID_19

project = get_project(model, region)

# Run a model manually.
# run_project_locally(project, run_scenarios=False)

# Run a calibration
project.calibrate(max_seconds=60, chain_idx=1, num_chains=1)
