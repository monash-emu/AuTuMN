"""
Entry point for PyCharm users to run an application
"""

from autumn.settings import Models, Region
from autumn.tools.project import get_project, run_project_locally

region = Region.BANGLADESH
model = Models.SM_SIR

project = get_project(model, region)

# Run a model manually.
run_project_locally(project, run_scenarios=False)

# Run a calibration
# project.calibrate(max_seconds=20, chain_idx=1, num_chains=1)
