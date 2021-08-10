from autumn.tools.project import Project, ParameterSet, TimeSeriesSet, build_rel_path
from autumn.tools.calibration import Calibration
from autumn.tools.calibration.priors import UniformPrior
from autumn.tools.calibration.targets import NormalTarget, get_dispersion_priors_for_gaussian_targets
from autumn.models.covid_19 import base_params, build_model
from autumn.settings import Region, Models
from autumn.projects.covid_19.calibration import COVID_GLOBAL_PRIORS

# Load and configure model parameters.
default_path = build_rel_path("params/default.yml")
baseline_params = base_params.update(default_path)
param_set = ParameterSet(baseline=baseline_params, scenarios=[])

ts_set = TimeSeriesSet.from_file(build_rel_path("timeseries.json"))
notifications_ts = ts_set.get("notifications").truncate_start_time(300)
infection_deaths_ts = ts_set.get("infection_deaths").truncate_start_time(300).downsample(7)
targets = [
    NormalTarget(notifications_ts),
    NormalTarget(infection_deaths_ts),
]

priors = [
    # Global COVID priors
    *COVID_GLOBAL_PRIORS,
    # Dispersion parameters based on targets
    *get_dispersion_priors_for_gaussian_targets(targets),
    # Regional parameters
    UniformPrior("contact_rate", [0.015, 0.03]),
    # Detection
    UniformPrior("testing_to_detection.assumed_cdr_parameter", [0.02, 0.06]),
    UniformPrior("voc_emergence.delta.contact_rate_multiplier", [2., 2.6]),
    UniformPrior("contact_tracing.assumed_trace_prop", [0.1, 0.3]),
]

calibration = Calibration(priors, targets)

# FIXME: Replace with flexible Python plot request API.
import json

plot_spec_filepath = build_rel_path("timeseries.json")
with open(plot_spec_filepath) as f:
    plot_spec = json.load(f)

project = Project(
    Region.INDONESIA, Models.COVID_19, build_model, param_set, calibration, plots=plot_spec
)
