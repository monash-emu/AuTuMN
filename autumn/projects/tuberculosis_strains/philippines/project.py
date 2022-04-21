from autumn.models.tuberculosis_strains import base_params, build_model
from autumn.settings import Models, Region
from autumn.tools.calibration import Calibration
from autumn.tools.calibration.priors import UniformPrior
from autumn.tools.calibration.targets import NormalTarget
from autumn.tools.project import ParameterSet, Project, build_rel_path, load_timeseries

# Load and configure model parameters.
baseline_params = base_params.update(build_rel_path("params/default.yml"))
scenario_params = [baseline_params.update(build_rel_path(f"params/scenario-1.yml"))]
param_set = ParameterSet(baseline=baseline_params, scenarios=scenario_params)
ts_set = load_timeseries(build_rel_path("timeseries.json"))
prev_inf_ts = ts_set["prevalence_infectious"]
targets = [NormalTarget(prev_inf_ts, time_weights=list(range(1, len(prev_inf_ts) + 1)))]
priors = [
    UniformPrior("contact_rate", [0.0, 200.0]),
    UniformPrior("initial_infectious_population", [1.0e4, 1.0e6]),
]
calibration = Calibration(priors, targets)

# FIXME: Replace with flexible Python plot request API.
import json

plot_spec_filepath = build_rel_path("timeseries.json")
with open(plot_spec_filepath) as f:
    plot_spec = json.load(f)

project = Project(
    Region.PHILIPPINES, Models.TBS, build_model, param_set, calibration, plots=plot_spec
)
