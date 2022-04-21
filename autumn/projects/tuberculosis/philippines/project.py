from autumn.models.tuberculosis import base_params, build_model
from autumn.settings import Models, Region
from autumn.tools.calibration import Calibration
from autumn.tools.calibration.priors import UniformPrior
from autumn.tools.calibration.targets import NormalTarget
from autumn.tools.project import ParameterSet, Project, build_rel_path, load_timeseries

# Load and configure model parameters.
default_path = build_rel_path("params/default.yml")
baseline_params = base_params.update(default_path)
param_set = ParameterSet(baseline=baseline_params)

# Load and configure calibration targets
ts_set = load_timeseries(build_rel_path("timeseries.json"))
prev_ts = ts_set["prevalence_infectious"]
targets = [NormalTarget(prev_ts, time_weights=list(range(1, len(prev_ts) + 1)))]
priors = [UniformPrior("contact_rate", [0.025, 0.05])]
calibration = Calibration(priors, targets)


# FIXME: Replace with flexible Python plot request API.
import json

plot_spec_filepath = build_rel_path("timeseries.json")
with open(plot_spec_filepath) as f:
    plot_spec = json.load(f)


project = Project(
    Region.PHILIPPINES, Models.TB, build_model, param_set, calibration, plots=plot_spec
)
