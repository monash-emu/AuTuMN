import json

from autumn.core.project import Project, ParameterSet, build_rel_path,\
    get_all_available_scenario_paths
from autumn.calibration import Calibration

from autumn.models.sm_sir import base_params, build_model
from autumn.settings import Region, Models
from autumn.projects.sm_sir.common import get_WPRO_priors, get_tartgets

# Load and configure model parameters.
mle_path = build_rel_path("params/mle-params.yml")
baseline_params = base_params.update(build_rel_path("params/baseline.yml")).update(
    mle_path, calibration_format=True
)
scenario_dir_path = build_rel_path("params/")
scenario_paths = get_all_available_scenario_paths(scenario_dir_path)

scenario_params = [baseline_params.update(p) for p in scenario_paths]
param_set = ParameterSet(baseline=baseline_params, scenarios=scenario_params)

# Load and configure calibration settings
calibration_start_time = param_set.baseline.to_dict()["time"]["start"]

priors = get_WPRO_priors()

country = "malaysia"
region = "malaysia"

targets = get_tartgets(calibration_start_time, country, region)

calibration = Calibration(
    priors=priors, targets=targets, random_process=None, metropolis_init="current_params"
)

plot_spec_filepath = build_rel_path("timeseries.json")
with open(plot_spec_filepath) as f:
    plot_spec = json.load(f)

# Create and register the project
project = Project(
    Region.MALAYSIA, Models.SM_SIR, build_model, param_set, calibration, plots=plot_spec
)