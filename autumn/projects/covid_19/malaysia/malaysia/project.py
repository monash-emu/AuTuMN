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
malaysia_path = build_rel_path("../malaysia/params/default.yml")
default_path = build_rel_path("params/default.yml")
scenario_paths = [build_rel_path(f"params/scenario-{i}.yml") for i in range(1, 16)]
mle_path = build_rel_path("params/mle-params.yml")
baseline_params = (
    base_params.update(malaysia_path).update(default_path).update(mle_path, calibration_format=True)
)
scenario_params = [baseline_params.update(p) for p in scenario_paths]
param_set = ParameterSet(baseline=baseline_params, scenarios=scenario_params)

ts_set = TimeSeriesSet.from_file(build_rel_path("timeseries.json"))
notifications_ts = ts_set.get("notifications").truncate_start_time(210)
icu_occupancy_ts = ts_set.get("icu_occupancy").truncate_start_time(210).downsample(7)
infection_deaths_ts = ts_set.get("infection_deaths").truncate_start_time(210).downsample(7)
targets = [
    NormalTarget(notifications_ts),
    NormalTarget(icu_occupancy_ts),
    NormalTarget(infection_deaths_ts),
]

priors = [
    # Global COVID priors
    *COVID_GLOBAL_PRIORS,
    # Dispersion parameters based on targets
    *get_dispersion_priors_for_gaussian_targets(targets),
    # Regional parameters
    UniformPrior("contact_rate", [0.015, 0.06]),
    UniformPrior("infectious_seed", [30.0, 200.0]),
    # Detection
    UniformPrior("testing_to_detection.assumed_cdr_parameter", [0.02, 0.1]),
    # Microdistancing
    UniformPrior("mobility.microdistancing.behaviour.parameters.upper_asymptote", [0.05, 0.4]),
    # Health system-related
    UniformPrior("clinical_stratification.props.hospital.multiplier", [0.7, 1.5]),
    UniformPrior("clinical_stratification.icu_prop", [0.12, 0.25]),
    UniformPrior("clinical_stratification.non_sympt_infect_multiplier", [0.15, 0.4]),
    UniformPrior("clinical_stratification.props.symptomatic.multiplier", [0.01, 1.5]),
    BetaPrior("vaccination.vacc_prop_prevent_infection", mean=0.7, ci=[0.5, 0.9], sampling="lhs"),
    UniformPrior("vaccination.overall_efficacy", [0.0, 1.0], sampling="lhs"),
    UniformPrior("vaccination.coverage_override", [0.0, 1.0], sampling="lhs"),
    UniformPrior("voc_emergence.contact_rate_multiplier", [1.0, 2.75]),
    UniformPrior("voc_emergence.start_time", [300, 400]),
]


calibration = Calibration(priors, targets)

# FIXME: Replace with flexible Python plot request API.
import json

plot_spec_filepath = build_rel_path("timeseries.json")
with open(plot_spec_filepath) as f:
    plot_spec = json.load(f)


project = Project(
    Region.MALAYSIA, Models.COVID_19, build_model, param_set, calibration, plots=plot_spec
)
