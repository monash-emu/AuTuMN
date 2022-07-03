from autumn.core.project import (
    Project,
    ParameterSet,
    load_timeseries,
    build_rel_path,
    use_tuned_proposal_sds,
)
from autumn.calibration import Calibration
from autumn.calibration.priors import UniformPrior
from autumn.calibration.targets import (
    NormalTarget,
    get_dispersion_priors_for_gaussian_targets,
)
from autumn.models.tb_dynamics import base_params, build_model
from autumn.settings import Region, Models

# Load and configure model parameters.
default_path = build_rel_path("params/default.yml")
baseline_params = base_params.update(default_path)
ts_set = load_timeseries(build_rel_path("timeseries.json"))

param_set = ParameterSet(baseline=baseline_params, scenarios = [])

targets = [
     NormalTarget(ts_set["population_size"], stdev=2500.0),
]

priors = [
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
