import numpy as np

from autumn.tools.calibration.proposal_tuning import perform_all_params_proposal_tuning
from autumn.tools.project import Project, ParameterSet, TimeSeriesSet, build_rel_path, get_all_available_scenario_paths, \
    use_tuned_proposal_sds
from autumn.tools.calibration import Calibration
from autumn.tools.calibration.priors import UniformPrior, BetaPrior, TruncNormalPrior
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
scenario_paths = [build_rel_path(f"params/scenario-{i}.yml") for i in range(5, 11)]
mle_path = build_rel_path("params/mle-params.yml")
baseline_params = (
    base_params.update(malaysia_path).update(default_path).update(mle_path, calibration_format=True)
)
scenario_params = [baseline_params.update(p) for p in scenario_paths]
param_set = ParameterSet(baseline=baseline_params, scenarios=scenario_params)

ts_set = TimeSeriesSet.from_file(build_rel_path("timeseries.json"))
notifications_ts = ts_set.get("notifications").truncate_start_time(210)
icu_occupancy_ts = ts_set.get("icu_occupancy").truncate_start_time(210).moving_average(window=7).downsample(step=7)
infection_deaths_ts = ts_set.get("infection_deaths").truncate_start_time(210).moving_average(window=7).downsample(step=7)
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
    UniformPrior("contact_rate", [0.015, 0.04]),
    UniformPrior("infectious_seed", [600.0, 950.0]),
    # Detection
    UniformPrior("testing_to_detection.assumed_cdr_parameter", [0.003, 0.015]),
    # Microdistancing
    UniformPrior("mobility.microdistancing.behaviour.parameters.upper_asymptote", [0.175, 0.3]),
    # Health system-related
    UniformPrior("clinical_stratification.icu_prop", [0.05, 0.15]),
    UniformPrior("clinical_stratification.non_sympt_infect_multiplier", [0.15, 0.6]),
    UniformPrior("clinical_stratification.props.symptomatic.multiplier", [0.8, 1.5]),
    UniformPrior("infection_fatality.multiplier", [0.1, 1.0]),
    #VoC parameters
    UniformPrior("voc_emergence.alpha_beta.contact_rate_multiplier", [1.3, 1.7]),
    UniformPrior("voc_emergence.alpha_beta.start_time", [300, 400]),
    UniformPrior("voc_emergence.delta.contact_rate_multiplier", [2.6, 2.85]),
    UniformPrior("voc_emergence.delta.start_time", [468, 485]),
]

# Load proposal sds from yml file
use_tuned_proposal_sds(priors, build_rel_path("proposal_sds.yml"))

calibration = Calibration(priors, targets)

# FIXME: Replace with flexible Python plot request API.
import json

plot_spec_filepath = build_rel_path("timeseries.json")
with open(plot_spec_filepath) as f:
    plot_spec = json.load(f)


project = Project(
    Region.MALAYSIA, Models.COVID_19, build_model, param_set, calibration, plots=plot_spec
)

#perform_all_params_proposal_tuning(project, calibration, priors, n_points=50, relative_likelihood_reduction=0.2)
