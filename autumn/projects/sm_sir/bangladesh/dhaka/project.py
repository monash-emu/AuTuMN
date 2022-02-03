import json

from autumn.tools.project import Project, ParameterSet, load_timeseries, build_rel_path
from autumn.tools.calibration import Calibration
from autumn.tools.calibration.priors import UniformPrior
from autumn.tools.calibration.targets import NormalTarget
from autumn.models.sm_sir import base_params, build_model
from autumn.settings import Region, Models

# Load and configure model parameters.
baseline_params = base_params.update(build_rel_path("params/baseline.yml"))
param_set = ParameterSet(baseline=baseline_params, scenarios=[])

# Load and configure calibration settings.
ts_set = load_timeseries(build_rel_path("timeseries.json"))
priors = [UniformPrior("contact_rate", (0.1, 0.2)), UniformPrior("infectious_seed", (1, 400))]

targets = [
    NormalTarget(data=ts_set["notifications"]),
    NormalTarget(data=ts_set["infection_deaths"]),
]

calibration = Calibration(
    priors=priors, targets=targets, random_process=None, metropolis_init="current_params"
)

plot_spec_filepath = build_rel_path("timeseries.json")
with open(plot_spec_filepath) as f:
    plot_spec = json.load(f)

# Create and register the project
project = Project(Region.DHAKA, Models.SM_SIR, build_model, param_set, calibration, plots=plot_spec)
