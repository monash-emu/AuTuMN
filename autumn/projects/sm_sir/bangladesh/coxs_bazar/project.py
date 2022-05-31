import json

from autumn.core.project import Project, ParameterSet, load_timeseries, build_rel_path
from autumn.calibration import Calibration
from autumn.calibration.priors import UniformPrior
from autumn.calibration.targets import NormalTarget
from autumn.models.sm_sir import base_params, build_model
from autumn.settings import Region, Models

# Load and configure model parameters
baseline_params = base_params.update(build_rel_path("params/baseline.yml"))
param_set = ParameterSet(baseline=baseline_params, scenarios=[])

# Load and configure calibration settings
ts_set = load_timeseries(build_rel_path("timeseries.json"))
priors = [
    UniformPrior("contact_rate", (0.03, 0.12)),
    UniformPrior("voc_emergence.omicron.new_voc_seed.start_time", (690.0, 720.0)),
    UniformPrior("testing_to_detection.assumed_cdr_parameter", (0.006, 0.016)),
    UniformPrior("hospital_stay.hospital_all.parameters.mean", (10., 25.)),
    UniformPrior("voc_emergence.omicron.contact_rate_multiplier", (1.2, 1.5)),
]

calibration_start_time = param_set.baseline.to_dict()["time"]["start"]
notifications_ts = ts_set["notifications"].loc[calibration_start_time:]
hospital_ts = ts_set["hospital_occupancy"].loc[calibration_start_time:]
icu_ts = ts_set["icu_occupancy"].loc[calibration_start_time:]
deaths_ts = ts_set["infection_deaths"].loc[calibration_start_time:]

targets = [
    NormalTarget(notifications_ts),
    NormalTarget(hospital_ts),
    # NormalTarget(icu_ts),
    # NormalTarget(deaths_ts),
]

calibration = Calibration(
    priors=priors, targets=targets, random_process=None, metropolis_init="current_params"
)

plot_spec_filepath = build_rel_path("timeseries.json")
with open(plot_spec_filepath) as f:
    plot_spec = json.load(f)

# Create and register the project
project = Project(
    Region.COXS_BAZAR, Models.SM_SIR, build_model, param_set, calibration, plots=plot_spec
)
