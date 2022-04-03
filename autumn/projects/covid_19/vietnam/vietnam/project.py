from autumn.tools.project import Project, ParameterSet, load_timeseries, build_rel_path, get_all_available_scenario_paths
from autumn.runners.calibration import Calibration
from autumn.runners.calibration.priors import UniformPrior
from autumn.runners.calibration.targets import (
    NormalTarget,
)
from autumn.models.covid_19 import base_params, build_model
from autumn.settings import Region, Models

from autumn.projects.covid_19.calibration import COVID_GLOBAL_PRIORS


# Load and configure model parameters.
default_path = build_rel_path("params/default.yml")
scenario_dir_path = build_rel_path("params/")
scenario_paths = get_all_available_scenario_paths(scenario_dir_path)
mle_path = build_rel_path("params/mle-params.yml")
baseline_params = base_params.update(default_path).update(mle_path, calibration_format=True)
scenario_params = [baseline_params.update(p) for p in scenario_paths]
param_set = ParameterSet(baseline=baseline_params, scenarios=scenario_params)

ts_set = load_timeseries(build_rel_path("timeseries.json"))
cutoff_time = 487  # 1 May 2021
notifications_ts = ts_set["notifications"].loc[cutoff_time:]
infection_deaths_ts = ts_set["infection_deaths"].loc[cutoff_time:]
targets = [
    NormalTarget(notifications_ts),
    NormalTarget(infection_deaths_ts),
]

priors = [
    # Global COVID priors
    *COVID_GLOBAL_PRIORS,
    UniformPrior("infectious_seed", [1, 50]),
    UniformPrior("contact_rate", [0.02, 0.03]),
    UniformPrior("testing_to_detection.assumed_cdr_parameter", [0.01, 0.1]),
]


calibration = Calibration(priors, targets)

# FIXME: Replace with flexible Python plot request API.
import json

plot_spec_filepath = build_rel_path("timeseries.json")
with open(plot_spec_filepath) as f:
    plot_spec = json.load(f)


project = Project(
    Region.VIETNAM, Models.COVID_19, build_model, param_set, calibration, plots=plot_spec
)
