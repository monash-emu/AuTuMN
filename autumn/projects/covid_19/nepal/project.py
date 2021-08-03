from autumn.tools.project import Project, ParameterSet, TimeSeriesSet, build_rel_path
from autumn.tools.calibration import Calibration
from autumn.tools.calibration.priors import UniformPrior, BetaPrior
from autumn.tools.calibration.targets import (
    NormalTarget,
    get_dispersion_priors_for_gaussian_targets,
)
from autumn.models.covid_19 import base_params, build_model
from autumn.settings import Region, Models

from autumn.projects.covid_19.calibration import COVID_GLOBAL_PRIORS


# Load and configure model parameters.
default_path = build_rel_path("params/default.yml")
baseline_params = base_params.update(default_path)
param_set = ParameterSet(baseline=baseline_params, scenarios=[])

ts_set = TimeSeriesSet.from_file(build_rel_path("timeseries.json"))
notifications_ts = ts_set.get("notifications").truncate_start_time(210)
infection_deaths_ts = ts_set.get("infection_deaths").truncate_start_time(210).downsample(7)
targets = [
    NormalTarget(notifications_ts),
    NormalTarget(infection_deaths_ts),
]

priors = [
    # Global COVID priors
    *COVID_GLOBAL_PRIORS,
    # Dispersion parameters based on targets
    *get_dispersion_priors_for_gaussian_targets(targets),
    UniformPrior("contact_rate", [0.015, 0.03]),
    UniformPrior("infectious_seed", [50.0, 200.0]),
    UniformPrior("testing_to_detection.assumed_cdr_parameter", [0.03, 0.15]),
    UniformPrior("mobility.microdistancing.behaviour.parameters.upper_asymptote", [0.05, 0.4]),
    UniformPrior("clinical_stratification.non_sympt_infect_multiplier", [0.15, 0.4]),
    BetaPrior("vaccination.vacc_prop_prevent_infection", mean=0.7, ci=[0.5, 0.9], sampling="lhs"),
    UniformPrior("vaccination.overall_efficacy", [0.0, 1.0], sampling="lhs"),
    UniformPrior("voc_emergence.voc_strain(0).voc_components.contact_rate_multiplier", [1.2, 2.1]),
    UniformPrior("voc_emergence.voc_strain(0).voc_components.start_time", [370, 400]),
]


calibration = Calibration(priors, targets)

# FIXME: Replace with flexible Python plot request API.
import json

plot_spec_filepath = build_rel_path("timeseries.json")
with open(plot_spec_filepath) as f:
    plot_spec = json.load(f)


project = Project(
    Region.NEPAL, Models.COVID_19, build_model, param_set, calibration, plots=plot_spec
)
