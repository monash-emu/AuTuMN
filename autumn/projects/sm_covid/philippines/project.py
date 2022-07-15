from autumn.core.project import (
    Project,
    ParameterSet,
    load_timeseries,
    build_rel_path,
    # use_tuned_proposal_sds,
    get_all_available_scenario_paths,
)
from autumn.calibration import Calibration
from autumn.calibration.priors import UniformPrior
from autumn.calibration.targets import NegativeBinomialTarget
from autumn.models.sm_covid import base_params, build_model
from autumn.model_features.random_process import set_up_random_process
from autumn.settings import Region, Models


# Load and configure model parameters.
mle_path = build_rel_path("params/mle-params.yml")
baseline_params = base_params.update(build_rel_path("params/baseline.yml")).update(
    mle_path, calibration_format=True
)

scenario_dir_path = build_rel_path("params/")
scenario_paths = get_all_available_scenario_paths(scenario_dir_path)

scenario_params = [baseline_params.update(p) for p in scenario_paths]
param_set = ParameterSet(baseline=baseline_params, scenarios=scenario_params)


# Load and configure calibration settings.
ts_set = load_timeseries(build_rel_path("timeseries.json"))
priors = [
    UniformPrior("contact_rate", [0.03, 0.20]),  
    UniformPrior("infectious_seed_time", [30, 90]),  
    UniformPrior("age_stratification.ifr.multiplier", [.5, 1.5]),

    UniformPrior("infection_deaths_dispersion_param", [5, 15]),  # greater values lead to a more skewed likelihood
]

model_end_time = baseline_params.to_dict()["time"]["end"]
targets = [
    NegativeBinomialTarget(data=ts_set["infection_deaths"].loc[:model_end_time], dispersion_param=7.),  # dispersion param from Watson et al. Lancet ID
]

if baseline_params.to_dict()["activate_random_process"]:
    rp_params = baseline_params.to_dict()["random_process"]
    rp = set_up_random_process(rp_params["time"]["start"], rp_params["time"]["end"], rp_params["order"], rp_params["time"]["step"])

    # rp = None  # use this when tuning proposal jumping steps
else:
    rp = None

calibration = Calibration(
    priors=priors, targets=targets, random_process=rp, metropolis_init="current_params"
)

# FIXME: Replace with flexible Python plot request API.
import json

plot_spec_filepath = build_rel_path("timeseries.json")
with open(plot_spec_filepath) as f:
    plot_spec = json.load(f)


# Create and register the project.
project = Project(Region.PHILIPPINES, Models.SM_COVID, build_model, param_set, calibration, plots=plot_spec)


# from autumn.calibration._tuning import perform_all_params_proposal_tuning
# perform_all_params_proposal_tuning(project, calibration, priors, n_points=20, relative_likelihood_reduction=0.2)
