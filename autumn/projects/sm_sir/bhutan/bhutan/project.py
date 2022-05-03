import json
from autumn.tools.project import Project, ParameterSet, load_timeseries, build_rel_path, get_all_available_scenario_paths
from autumn.tools.calibration import Calibration
from autumn.tools.calibration.priors import UniformPrior
from autumn.tools.calibration.targets import NormalTarget
from autumn.models.sm_sir import base_params, build_model
from autumn.settings import Region, Models

# Load and configure model parameters.
mle_path = build_rel_path("params/mle-params.yml")
baseline_params = base_params.update(build_rel_path("params/baseline.yml")).update(
    mle_path, calibration_format=True
)
# scenario_dir_path = build_rel_path("params/")
# scenario_paths = get_all_available_scenario_paths(scenario_dir_path)

# scenario_params = [baseline_params.update(p) for p in scenario_paths]
param_set = ParameterSet(baseline=baseline_params)

# Load and configure calibration settings
calibration_start_time = param_set.baseline.to_dict()["time"]["start"]
ts_set = load_timeseries(build_rel_path("timeseries.json"))
notifications_ts = ts_set["notifications"].loc[calibration_start_time:]

targets = [
    NormalTarget(notifications_ts),
]

priors = [
    UniformPrior("contact_rate", (0.2, 0.4)),
    UniformPrior("sojourns.latent.total_time", (1., 4.)),
    UniformPrior("testing_to_detection.assumed_cdr_parameter", (0.0005, 0.005)),
    UniformPrior("infectious_seed", (50., 250.))
]

calibration = Calibration(
    priors=priors, targets=targets, random_process=None, metropolis_init="current_params"
)


plot_spec_filepath = build_rel_path("timeseries.json")
with open(plot_spec_filepath) as f:
    plot_spec = json.load(f)


# Create and register the project
project = Project(
    Region.BHUTAN, Models.SM_SIR, build_model, param_set, calibration, plots=plot_spec
)