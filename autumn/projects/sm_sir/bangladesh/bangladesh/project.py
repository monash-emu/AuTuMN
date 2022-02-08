import json

from autumn.tools.project import Project, ParameterSet, load_timeseries, build_rel_path
from autumn.tools.calibration import Calibration
from autumn.tools.calibration.priors import UniformPrior
from autumn.tools.calibration.targets import NormalTarget
from autumn.models.sm_sir import base_params, build_model
from autumn.settings import Region, Models

# Load and configure model parameters
baseline_params = base_params.update(build_rel_path("params/baseline.yml"))
param_set = ParameterSet(baseline=baseline_params, scenarios=[])

# Load and configure calibration settings.
ts_set = load_timeseries(build_rel_path("timeseries.json"))
calibration_start_time = param_set.baseline.to_dict()["time"]["start"]
notifications_ts = ts_set["notifications"].loc[calibration_start_time:]

priors = [
    UniformPrior("contact_rate", (0.08, 0.18)),
    UniformPrior("testing_to_detection.assumed_cdr_parameter", (0.003, 0.012)),
    UniformPrior("voc_emergence.omicron.new_voc_seed.start_time", (640., 680.)),
    UniformPrior("voc_emergence.omicron.contact_rate_multiplier", (0.5, 1.))]

targets = [
    NormalTarget(notifications_ts),
]

calibration = Calibration(priors=priors, targets=targets, random_process=None, metropolis_init="current_params")

plot_spec_filepath = build_rel_path("timeseries.json")
with open(plot_spec_filepath) as f:
    plot_spec = json.load(f)

# Create and register the project
project = Project(Region.BANGLADESH, Models.SM_SIR, build_model, param_set, calibration, plots=plot_spec)
