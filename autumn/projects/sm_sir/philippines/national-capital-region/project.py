from autumn.tools.project import (
    Project,
    ParameterSet,
    load_timeseries,
    build_rel_path,
    use_tuned_proposal_sds,
)
from autumn.tools.calibration import Calibration
from autumn.tools.calibration.priors import UniformPrior
from autumn.tools.calibration.targets import NormalTarget
from autumn.models.sm_sir import base_params, build_model, set_up_random_process
from autumn.settings import Region, Models

# Load and configure model parameters.
mle_path = build_rel_path("params/mle-params.yml")
baseline_params = base_params.update(build_rel_path("params/baseline.yml")).update(
    mle_path, calibration_format=True
)
param_set = ParameterSet(baseline=baseline_params, scenarios=[])

# Load and configure calibration settings.
ts_set = load_timeseries(build_rel_path("timeseries.json"))
priors = [
    UniformPrior("contact_rate", [0.04, 0.08]),
    UniformPrior("sojourns.active.total_time", [5, 10]),
    UniformPrior("infectious_seed", [1, 400]),
    UniformPrior("testing_to_detection.assumed_cdr_parameter", [0.01, 0.05]),
    # Pre-existing immunity
    UniformPrior(
        "immunity_stratification.prop_immune", [0.8, 0.9]
    ),  # 80% of total NCR pop with at least one dose by 29 Dec
    UniformPrior(
        "immunity_stratification.prop_high_among_immune", [0.05, 0.20]
    ),  # Among people with at least one dose, 5% had received a booster by 29 Dec.
    # Hospital-related
    UniformPrior("hospital_prop_multiplier", [0.2, 1.]),
    UniformPrior("time_from_onset_to_event.hospitalisation.parameters.mean", [2.0, 7.0]),
    UniformPrior("prop_icu_among_hospitalised", [0.05, 0.20]),
    UniformPrior("hospital_stay.hospital_all.parameters.mean", [2.0, 5.0]),
    UniformPrior("hospital_stay.icu.parameters.mean", [3.0, 10.0]),
]

targets = [
    NormalTarget(data=ts_set["icu_occupancy"].loc[725:]),
    NormalTarget(data=ts_set["hospital_occupancy"].loc[725:])
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


# from autumn.tools.calibration.proposal_tuning import perform_all_params_proposal_tuning
# perform_all_params_proposal_tuning(project, calibration, priors, n_points=50, relative_likelihood_reduction=0.2)
