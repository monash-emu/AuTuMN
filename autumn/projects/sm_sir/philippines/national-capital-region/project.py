from autumn.tools.project import (
    Project,
    ParameterSet,
    load_timeseries,
    build_rel_path,
    use_tuned_proposal_sds,
    get_all_available_scenario_paths,
)
from autumn.runners import Calibration
from autumn.calibration.priors import UniformPrior
from autumn.calibration import NormalTarget
from autumn.models.sm_sir import base_params, build_model, set_up_random_process
from autumn.settings import Region, Models

# Load and configure model parameters.
mle_path = build_rel_path("params/mle-params.yml")
scenario_dir_path = build_rel_path("params/")
scenario_paths = get_all_available_scenario_paths(scenario_dir_path)

baseline_params = base_params.update(build_rel_path("params/baseline.yml")).update(
    mle_path, calibration_format=True
)
scenario_params = [baseline_params.update(p) for p in scenario_paths]
param_set = ParameterSet(baseline=baseline_params, scenarios=scenario_params)

# Load and configure calibration settings.
ts_set = load_timeseries(build_rel_path("timeseries.json"))
priors = [
    UniformPrior("contact_rate", [0.07, 0.20]),
    UniformPrior("sojourns.active.total_time", [4, 10]),
    UniformPrior("infectious_seed", [1, 400]),
    UniformPrior("testing_to_detection.assumed_cdr_parameter", [0.005, 0.015]),
    # Vaccine-induced immunity
    UniformPrior(
        "immunity_stratification.infection_risk_reduction.low", [0.038, 0.438]
    ),
    UniformPrior(
        "immunity_stratification.infection_risk_reduction.high", [0.438, 0.6]
    ),
    UniformPrior(
        "age_stratification.prop_hospital.source_immunity_protection.low", [0.488, 0.807]
    ),
    UniformPrior(
        "age_stratification.prop_hospital.source_immunity_protection.high", [.85, 0.95]
    ),
    # Hospital-related
    UniformPrior("age_stratification.prop_hospital.multiplier", [0.5, 1.5]),
    UniformPrior("time_from_onset_to_event.hospitalisation.parameters.mean", [2.0, 7.0]),
    UniformPrior("prop_icu_among_hospitalised", [0.05, 0.20]),
    UniformPrior("hospital_stay.hospital_all.parameters.mean", [2.0, 8.0]),
    UniformPrior("hospital_stay.icu.parameters.mean", [3.0, 10.0]),
]

new_target_set = load_timeseries(build_rel_path("new_targets.json"))

targets = [
    NormalTarget(data=new_target_set["icu_occupancy"].loc[725:]),
    NormalTarget(data=new_target_set["hospital_occupancy"].loc[725:]),
    NormalTarget(data=ts_set["notifications"].loc[741:755]),  # peak notifications
]

if baseline_params.to_dict()["activate_random_process"]:
    time_params = baseline_params.to_dict()["time"]
    rp = set_up_random_process(time_params["start"], time_params["end"])

    # rp = None  # use this when tuning proposal jumping steps
else:
    rp = None

# Load proposal sds from yml file
use_tuned_proposal_sds(priors, build_rel_path("proposal_sds.yml"))

calibration = Calibration(
    priors=priors, targets=targets, random_process=rp, metropolis_init="current_params"
)

# FIXME: Replace with flexible Python plot request API.
import json

plot_spec_filepath = build_rel_path("timeseries.json")
with open(plot_spec_filepath) as f:
    plot_spec = json.load(f)


# Create and register the project.
project = Project(Region.NCR, Models.SM_SIR, build_model, param_set, calibration, plots=plot_spec)


# from autumn.calibration._tuning import perform_all_params_proposal_tuning
# perform_all_params_proposal_tuning(project, calibration, priors, n_points=20, relative_likelihood_reduction=0.2)
