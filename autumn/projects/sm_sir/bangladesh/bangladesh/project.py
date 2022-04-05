import json
from datetime import date
import numpy as np
import pandas as pd

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
late_hosp_admissions = ts_set["hospital_admissions"].loc[notifications_trunc_point:]
deaths_ts = ts_set["infection_deaths"].loc[targets_start:]
late_deaths = ts_set["infection_deaths"].loc[notifications_trunc_point:]


def find_increment(input_series):
    if type(input_series) == pd.Series:
        return input_series.diff()
    else:
        return np.append([np.nan], np.diff(input_series))


ts_set["notif_change"] = find_increment(ts_set["notifications"])


priors = [
    UniformPrior("contact_rate", (0.02, 0.1)),
    UniformPrior("testing_to_detection.assumed_cdr_parameter", (0.005, 0.015)),
    UniformPrior("voc_emergence.omicron.new_voc_seed.start_time", (500., 550.)),
    UniformPrior("voc_emergence.omicron.contact_rate_multiplier", (2.2, 3.5)),
    UniformPrior("age_stratification.cfr.multiplier", (0.01, 0.08)),
    UniformPrior("age_stratification.prop_hospital.multiplier", (0.01, 0.08)),
]

# Not sure about this
notif_change = notifications_ts.diff()
notif_change_smoothed = notif_change.rolling(window=7).mean().dropna()

targets = [
    NormalTarget(notifications_ts),
    NormalTarget(hospital_admissions_ts),
    NormalTarget(deaths_ts),
    NormalTarget(late_hosp_admissions),
    NormalTarget(late_deaths),
]

calibration = Calibration(priors=priors, targets=targets, random_process=None, metropolis_init="current_params")

plot_spec_filepath = build_rel_path("timeseries.json")
with open(plot_spec_filepath) as f:
    plot_spec = json.load(f)


def custom_build_model(param_set, build_options=None):
    model = build_model(param_set, build_options)
    model.request_function_output(
        "notif_change",
        find_increment,
        ["notifications"],
    )
    return model


# Create and register the project
project = Project(Region.BANGLADESH, Models.SM_SIR, custom_build_model, param_set, calibration, plots=plot_spec)
