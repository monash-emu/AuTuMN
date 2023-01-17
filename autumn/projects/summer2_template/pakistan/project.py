from pandas import Series

from autumn.core.project import Project, ParameterSet, build_rel_path, get_all_available_scenario_paths
from autumn.calibration import Calibration
from autumn.calibration.priors import UniformPrior
from autumn.calibration.targets import NormalTarget

from autumn.models.summer2_template import base_params, build_model
from autumn.settings import Region, Models

# Load and configure model parameters.
baseline_params = base_params.update(build_rel_path("params/baseline.yml"))
scenario_dir_path = build_rel_path("params/")
scenario_paths = get_all_available_scenario_paths(scenario_dir_path)

scenario_params = [baseline_params.update(p) for p in scenario_paths]
param_set = ParameterSet(baseline=baseline_params, scenarios=scenario_params)

# Create some dummy incidence data
incidence_data = Series(name="incidence", index=[50, 100], data=[20, 50])

targets = [
    NormalTarget(incidence_data, stdev=10.),
]

priors = [
    UniformPrior("contact_rate", (0.1, 0.3)),
]

calibration = Calibration(
    priors=priors, targets=targets, random_process=None, metropolis_init="current_params"
)
# Create and register the project
project = Project(
    Region.PAKISTAN, Models.SUMMER2_TEMPLATE, build_model, param_set, calibration
)