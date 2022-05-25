import numpy as np
from autumn.runners.calibration.proposal_tuning import perform_all_params_proposal_tuning
from autumn.tools.project import Project, ParameterSet, load_timeseries, build_rel_path, get_all_available_scenario_paths, \
    use_tuned_proposal_sds
from autumn.runners.calibration import Calibration
from autumn.runners.calibration.priors import UniformPrior, BetaPrior,TruncNormalPrior
from autumn.runners.calibration.targets import (
    NormalTarget,
    get_dispersion_priors_for_gaussian_targets,
)
from autumn.models.covid_19 import base_params, build_model
from autumn.settings import Region, Models
from autumn.projects.covid_19.sri_lanka.sri_lanka.scenario_builder import get_all_scenario_dicts

# Load and configure model parameters.
default_path = build_rel_path("params/default.yml")
#scenario_paths = [build_rel_path(f"params/scenario-{i}.yml") for i in range(7, 9)]
mle_path = build_rel_path("params/mle-params.yml")
baseline_params = base_params.update(default_path).update(mle_path, calibration_format=True)
all_scenario_dicts = get_all_scenario_dicts("LKA")
#scenario_params = [baseline_params.update(p) for p in scenario_paths]
scenario_params = [baseline_params.update(sc_dict) for sc_dict in all_scenario_dicts]
param_set = ParameterSet(baseline=baseline_params, scenarios=scenario_params)

ts_set = load_timeseries(build_rel_path("timeseries.json"))
notifications_ts = ts_set["notifications"].rolling(7).mean().loc[350::7]
death_ts = ts_set["infection_deaths"].loc[350:]
targets = [
    NormalTarget(notifications_ts),
    NormalTarget(death_ts),
]

priors = [
    # Dispersion parameters based on targets
    *get_dispersion_priors_for_gaussian_targets(targets),
    *get_dispersion_priors_for_gaussian_targets(targets),
    # Regional parameters
    UniformPrior("contact_rate", [0.024, 0.027]),
    UniformPrior("infectious_seed", [275.0, 450.0]),
    # Detection
    UniformPrior("testing_to_detection.assumed_cdr_parameter", [0.009, 0.025]),
    UniformPrior("infection_fatality.multiplier", [0.09, 0.13]),
    #VoC
    UniformPrior("voc_emergence.alpha_beta.start_time", [370, 410]),
    UniformPrior("voc_emergence.alpha_beta.contact_rate_multiplier", [3.2, 4.5]),
    UniformPrior("voc_emergence.delta.start_time", [475, 530]),
    UniformPrior("voc_emergence.delta.contact_rate_multiplier", [8.5, 11.5]),
]

# Load proposal sds from yml file
# use_tuned_proposal_sds(priors, build_rel_path("proposal_sds.yml"))

calibration = Calibration(priors, targets)

# FIXME: Replace with flexible Python plot request API.
import json

plot_spec_filepath = build_rel_path("timeseries.json")
with open(plot_spec_filepath) as f:
    plot_spec = json.load(f)


project = Project(
    Region.SRI_LANKA, Models.COVID_19, build_model, param_set, calibration, plots=plot_spec
)

#perform_all_params_proposal_tuning(project, calibration, priors, n_points=50, relative_likelihood_reduction=0.2)