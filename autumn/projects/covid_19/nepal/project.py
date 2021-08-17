from autumn.tools.project import Project, ParameterSet, TimeSeriesSet, build_rel_path
from autumn.tools.calibration import Calibration
from autumn.tools.calibration.priors import UniformPrior, TruncNormalPrior
from autumn.tools.calibration.targets import (
    NormalTarget,
    get_dispersion_priors_for_gaussian_targets,
)
from autumn.models.covid_19 import base_params, build_model
from autumn.settings import Region, Models
from autumn.projects.covid_19.calibration import COVID_GLOBAL_PRIORS

# FIXME: Replace with flexible Python plot request API.
import json

# Load and configure model parameters.
default_path = build_rel_path("params/default.yml")
mle_path = build_rel_path("params/mle-params.yml")
baseline_params = base_params.update(default_path).update(mle_path, calibration_format=True)
param_set = ParameterSet(baseline=baseline_params, scenarios=[])

ts_set = TimeSeriesSet.from_file(build_rel_path("timeseries.json"))
notifications_ts = ts_set.get("notifications").truncate_start_time(210)
seroprevalence_estimate = ts_set.get("proportion_seropositive")
targets = [
    NormalTarget(notifications_ts),
    NormalTarget(seroprevalence_estimate, stdev=0.1),
]

priors = [
    # Global COVID priors
    *COVID_GLOBAL_PRIORS,
    # Dispersion parameters based on targets
    *get_dispersion_priors_for_gaussian_targets(targets),
    UniformPrior("contact_rate", (0.015, 0.03)),
    UniformPrior("testing_to_detection.assumed_cdr_parameter", (0.03, 0.15)),
    UniformPrior("clinical_stratification.non_sympt_infect_multiplier", (0.15, 0.6)),
    TruncNormalPrior("voc_emergence.delta.contact_rate_multiplier", 2., 0.2, (1., 10.))
]

calibration = Calibration(priors, targets)

plot_spec_filepath = build_rel_path("timeseries.json")
with open(plot_spec_filepath) as f:
    plot_spec = json.load(f)

project = Project(
    Region.NEPAL, Models.COVID_19, build_model, param_set, calibration, plots=plot_spec
)
