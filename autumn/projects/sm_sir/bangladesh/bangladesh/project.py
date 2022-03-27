import json
from datetime import date

from autumn.models.sm_sir.parameters import BASE_DATE
from autumn.tools.project import Project, ParameterSet, load_timeseries, build_rel_path
from autumn.tools.calibration import Calibration
from autumn.tools.calibration.priors import UniformPrior
from autumn.tools.calibration.targets import NormalTarget
from autumn.models.sm_sir import base_params, build_model
from autumn.settings import Region, Models

# Load and configure model parameters
baseline_params = base_params.update(build_rel_path("params/baseline.yml"))
param_set = ParameterSet(baseline=baseline_params, scenarios=[])

# Load and configure calibration settings.
ts_set = load_timeseries(build_rel_path("timeseries.json"))
calibration_start_time = param_set.baseline.to_dict()["time"]["start"]

# Work out date truncation points
targets_start = (date(2021, 5, 15) - BASE_DATE).days
notifications_trunc_point = (date(2021, 12, 1) - BASE_DATE).days

# Get the actual targets
notifications_ts = ts_set["notifications"].loc[targets_start: notifications_trunc_point]
hospital_admissions_ts = ts_set["hospital_admissions"].loc[targets_start:]
deaths_ts = ts_set["infection_deaths"].loc[targets_start:]

priors = [
    UniformPrior("contact_rate", (0.07, 0.17)),
    UniformPrior("testing_to_detection.assumed_cdr_parameter", (0.005, 0.015)),
    UniformPrior("voc_emergence.omicron.new_voc_seed.start_time", (520., 570.)),
    UniformPrior("voc_emergence.omicron.contact_rate_multiplier", (0.5, 1.)),
    UniformPrior("age_stratification.cfr.multiplier", (0.01, 0.08)),
    UniformPrior("age_stratification.prop_hospital.multiplier", (0.01, 0.08)),
]

targets = [
    NormalTarget(notifications_ts),
    NormalTarget(hospital_admissions_ts),
    NormalTarget(deaths_ts),
]

calibration = Calibration(priors=priors, targets=targets, random_process=None, metropolis_init="current_params")

plot_spec_filepath = build_rel_path("timeseries.json")
with open(plot_spec_filepath) as f:
    plot_spec = json.load(f)

# Create and register the project
project = Project(Region.BANGLADESH, Models.SM_SIR, build_model, param_set, calibration, plots=plot_spec)
