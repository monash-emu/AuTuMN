from autumn.tools.project import Project, ParameterSet, TimeSeriesSet, build_rel_path
from autumn.tools.calibration import Calibration
from autumn.tools.calibration.priors import UniformPrior
from autumn.tools.calibration.targets import (
    NormalTarget,
    get_dispersion_priors_for_gaussian_targets,
)
from autumn.models.covid_19 import base_params, build_model
from autumn.settings import Region, Models

from autumn.projects.covid_19.calibration import COVID_GLOBAL_PRIORS


# Load and configure model parameters.
# Load and configure model parameters.
malaysia_path = build_rel_path("../malaysia/params/default.yml")
default_path = build_rel_path("params/default.yml")
baseline_params = base_params.update(malaysia_path).update(default_path)
param_set = ParameterSet(baseline=baseline_params)

ts_set = TimeSeriesSet.from_file(build_rel_path("timeseries.json"))
notifications_ts = ts_set.get("notifications")
targets = [NormalTarget(notifications_ts)]

priors = [
    # Global COVID priors
    *COVID_GLOBAL_PRIORS,
    # Dispersion parameters based on targets
    *get_dispersion_priors_for_gaussian_targets(targets),
    # Regional parameters
    UniformPrior("contact_rate", [0.08, 0.2]),
    # Health system-related
    UniformPrior("clinical_stratification.props.hospital.multiplier", [0.7, 1.3]),
    UniformPrior("sojourn.compartment_periods.icu_early", [5.0, 25.0]),
    UniformPrior("clinical_stratification.icu_prop", [0.12, 0.25]),
    # Detection
    UniformPrior("testing_to_detection.assumed_cdr_parameter", [0.05, 0.4]),
]

calibration = Calibration(priors, targets)

# FIXME: Replace with flexible Python plot request API.
import json

plot_spec_filepath = build_rel_path("timeseries.json")
with open(plot_spec_filepath) as f:
    plot_spec = json.load(f)

project = Project(
    Region.SABAH, Models.COVID_19, build_model, param_set, calibration, plots=plot_spec
)
