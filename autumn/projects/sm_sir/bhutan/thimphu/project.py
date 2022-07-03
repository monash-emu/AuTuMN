import json
from autumn.core.project import Project, ParameterSet, load_timeseries, build_rel_path
from autumn.calibration import Calibration
from autumn.calibration.priors import UniformPrior
from autumn.calibration.targets import NormalTarget
from autumn.models.sm_sir import base_params, build_model, set_up_random_process
from autumn.settings import Region, Models

# Load and configure model parameters
baseline_params = base_params.update(build_rel_path("params/baseline.yml"))
scenario_1_params = baseline_params.update(build_rel_path("params/scenario-1.yml"))
param_set = ParameterSet(baseline=baseline_params, scenarios=[scenario_1_params])

# Load and configure calibration settings
ts_set = load_timeseries(build_rel_path("timeseries.json"))
priors = [
    UniformPrior("contact_rate", [0.025, 0.05]),
]
targets = [
    NormalTarget(data=ts_set["notifications"]),
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


plot_spec_filepath = build_rel_path("timeseries.json")
with open(plot_spec_filepath) as f:
    plot_spec = json.load(f)


# Create and register the project
project = Project(
    Region.THIMPHU, Models.SM_SIR, build_model, param_set, calibration, plots=plot_spec
)
