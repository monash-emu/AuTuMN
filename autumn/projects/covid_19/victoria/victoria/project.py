import json

from autumn.tools.project import Project, ParameterSet, TimeSeriesSet, build_rel_path, get_all_available_scenario_paths
from autumn.tools.calibration import Calibration
from autumn.tools.calibration.priors import UniformPrior
from autumn.tools.calibration.targets import NormalTarget
from autumn.models.covid_19 import base_params, build_model
from autumn.settings import Region, Models

# Load and configure model parameters
default_path = build_rel_path("params/default.yml")
baseline_params = base_params.update(default_path)

scenario_dir_path = build_rel_path("params/")
scenario_paths = get_all_available_scenario_paths(scenario_dir_path)
scenario_params = [baseline_params.update(p) for p in scenario_paths]
param_set = ParameterSet(baseline=baseline_params, scenarios=scenario_params)

# Load and configure calibration settings
ts_set = TimeSeriesSet.from_file(build_rel_path("targets.secret.json"))
notifications = ts_set.get("notifications").truncate_start_time(455)
targets = [
    NormalTarget(notifications),
    # NormalTarget(infection_deaths)
]
priors = [
    UniformPrior("contact_rate", (0.05, 0.08), jumping_stdev=0.01),
    
]

calibration = Calibration(priors=priors, targets=targets)

plot_spec_filepath = build_rel_path("targets.secret.json")
with open(plot_spec_filepath) as f:
    plot_spec = json.load(f)

project = Project(Region.VICTORIA, Models.COVID_19, build_model, param_set, calibration, plots=plot_spec)
