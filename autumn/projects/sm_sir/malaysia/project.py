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
notifications_ts = ts_set["notifications"].loc[210:]
icu_occupancy_ts = ts_set["icu_occupancy"].rolling(7).mean().loc[210::7]
infection_deaths_ts = ts_set["infection_deaths"].loc[210:]

targets = [
    NormalTarget(notifications_ts),
    NormalTarget(infection_deaths_ts),
    NormalTarget(icu_occupancy_ts),
]

priors = [
    UniformPrior("contact_rate", [0.015, 0.04]),
]


if baseline_params.to_dict()["activate_random_process"]:
    time_params = baseline_params.to_dict()["time"]
    rp = set_up_random_process(time_params["start"], time_params["end"])

    # rp = None  # use this when tuning proposal jumping steps
else:
    rp = None

# Load proposal sds from yml file
# use_tuned_proposal_sds(priors, build_rel_path("proposal_sds.yml"))

calibration = Calibration(
    priors=priors, targets=targets, random_process=rp, metropolis_init="current_params"
)

# FIXME: Replace with flexible Python plot request API.
import json

plot_spec_filepath = build_rel_path("timeseries.json")
with open(plot_spec_filepath) as f:
    plot_spec = json.load(f)


# Create and register the project.
project = Project(
    Region.MALAYSIA, Models.SM_SIR, build_model, param_set, calibration, plots=plot_spec
)


# from autumn.tools.calibration.proposal_tuning import perform_all_params_proposal_tuning
# perform_all_params_proposal_tuning(project, calibration, priors, n_points=50, relative_likelihood_reduction=0.2)
