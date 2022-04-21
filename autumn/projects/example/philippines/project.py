from autumn.models.example import base_params, build_model
from autumn.settings import Models, Region
from autumn.tools.calibration import Calibration
from autumn.tools.calibration.priors import UniformPrior
from autumn.tools.calibration.targets import NormalTarget
from autumn.tools.project import ParameterSet, Project, build_rel_path, load_timeseries

# Load and configure model parameters
baseline_params = base_params.update(build_rel_path("params/baseline.yml"))
scenario_1_params = baseline_params.update(build_rel_path("params/scenario-1.yml"))
param_set = ParameterSet(baseline=baseline_params, scenarios=[scenario_1_params])

# Load and configure calibration settings
ts_set = load_timeseries(build_rel_path("timeseries.json"))
priors = [
    UniformPrior("contact_rate", [0.025, 0.05]),
    UniformPrior("recovery_rate", [0.9, 1.2]),
]
targets = [
    NormalTarget(data=ts_set["prevalence_infectious"]),
]
calibration = Calibration(priors=priors, targets=targets)

# Create and register the project
project = Project(
    Region.PHILIPPINES, Models.EXAMPLE, build_model, param_set, calibration
)
