from autumn.core.project import (
    Project,
    ParameterSet,
    load_timeseries,
    build_rel_path,
)
from autumn.calibration import Calibration
from autumn.calibration.priors import UniformPrior
from autumn.calibration.targets import NormalTarget
from autumn.models.tb_dynamics import base_params, build_model
from autumn.settings import Region, Models

# Load and configure model parameters.
default_path = build_rel_path("params/default.yml")
baseline_params = base_params.update(default_path)
ts_set = load_timeseries(build_rel_path("timeseries.json"))
param_set = ParameterSet(baseline=baseline_params, scenarios=[])

targets = [
    NormalTarget(ts_set["total_population"], stdev=2500.0),  
    NormalTarget(ts_set["notifications"], stdev=40.0),
]

priors = [
    UniformPrior("rr_infection_latent", [0.2, 0.5]),
    UniformPrior("rr_infection_recovered", [0.2, 1.0]),
    UniformPrior("contact_rate", [0.002, 0.01]),
    UniformPrior("infectious_seed", [300, 1000]),
]
calibration = Calibration(
    priors, targets, metropolis_init="current_params", metropolis_init_rel_step_size=0.1
)

project = Project(
    Region.KIRIBATI,
    Models.TBD,
    build_model,
    param_set,
    calibration,
)
