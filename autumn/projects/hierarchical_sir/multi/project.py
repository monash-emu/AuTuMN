from email.mime import base
from autumn.tools.project import Project, ParameterSet, load_timeseries, build_rel_path
from autumn.tools.calibration import Calibration
from autumn.tools.calibration.priors import UniformPrior
from autumn.tools.calibration.targets import NormalTarget
from autumn.models.hierarchical_sir import base_params, build_model
from autumn.settings import Region, Models

# Load and configure model parameters

baseline_params = base_params.update(build_rel_path("params/baseline.yml"))
param_set = ParameterSet(baseline=baseline_params)

# Load and configure calibration settings
ts_set = load_timeseries(build_rel_path("timeseries.json"))
priors = [
    UniformPrior("beta", [0.025, 0.05]),
    UniformPrior("gamma", [0.9, 1.2]),
]
targets = [
    NormalTarget(data=ts_set["incidence"]),
]
calibration = Calibration(priors=priors, targets=targets)

# Create and register the project
project = Project(Region.MULTI, Models.HIERARCHICAL_SIR, build_model, param_set, calibration)
