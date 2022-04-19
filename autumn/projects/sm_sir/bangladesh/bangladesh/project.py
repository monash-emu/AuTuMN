import json

from pandas import Series
from datetime import datetime

from autumn.tools.utils.utils import wrap_series_transform_for_ndarray
from autumn.settings.constants import COVID_BASE_DATETIME
from autumn.tools.project import (
    Project, ParameterSet, load_timeseries, build_rel_path, get_all_available_scenario_paths
)
from autumn.tools.calibration import Calibration
from autumn.tools.calibration.priors import UniformPrior
from autumn.tools.calibration.targets import NormalTarget, TruncNormalTarget
from autumn.models.sm_sir import base_params, build_model
from autumn.settings import Region, Models

# Load and configure model parameters
baseline_params = base_params.update(
    build_rel_path("params/baseline.yml")
).update(
    build_rel_path("params/mle-params.yml"), calibration_format=True,
)
scenario_dir_path = build_rel_path("params/")
scenario_paths = get_all_available_scenario_paths(scenario_dir_path)
scenario_params = [baseline_params.update(p) for p in scenario_paths]

param_set = ParameterSet(baseline=baseline_params, scenarios=scenario_params)

# Load and configure calibration settings.
ts_set = load_timeseries(build_rel_path("timeseries.json"))
calibration_start_time = param_set.baseline.to_dict()["time"]["start"]

# Work out date truncation points
targets_start = (datetime(2021, 5, 15) - COVID_BASE_DATETIME).days
notifications_trunc_point = (datetime(2021, 12, 1) - COVID_BASE_DATETIME).days
notif_change_start_point = (datetime(2022, 1, 29) - COVID_BASE_DATETIME).days
hospital_floor_start = (datetime(2022, 1, 1) - COVID_BASE_DATETIME).days

# Get the actual targets
notifications_ts = ts_set["notifications"].loc[targets_start: notifications_trunc_point]
hospital_admit_early_ts = ts_set["hospital_admissions"].loc[targets_start: hospital_floor_start]
hospital_admit_late_ts = ts_set["hospital_admissions"].loc[hospital_floor_start:]
deaths_ts = ts_set["infection_deaths"].loc[targets_start:]
late_deaths = ts_set["infection_deaths"].loc[notifications_trunc_point:]


# Build transformed target series
def get_roc(series: Series) -> Series:
    return series.rolling(7).mean().pct_change(7)


# Build targets for rate-of-change (roc) variables
roc_vars = ["notifications"]
roc_targets = []
for roc_var in roc_vars:
    new_ts = get_roc(ts_set[roc_var]).loc[notif_change_start_point:]
    new_ts.name = roc_var + '_roc'
    roc_targets.append(NormalTarget(new_ts))

targets = [
    NormalTarget(notifications_ts),
    NormalTarget(hospital_admit_early_ts),
    TruncNormalTarget(hospital_admit_late_ts, trunc_range=(20., 1e7)),
    NormalTarget(deaths_ts),
    NormalTarget(late_deaths)
] + roc_targets

priors = [
    UniformPrior("contact_rate", (0.02, 0.1)),
    UniformPrior("testing_to_detection.assumed_cdr_parameter", (0.005, 0.015)),
    UniformPrior("voc_emergence.omicron.new_voc_seed.start_time", (580., 680.)),
    UniformPrior("voc_emergence.omicron.contact_rate_multiplier", (2.2, 3.5)),
    UniformPrior("age_stratification.cfr.multiplier", (0.01, 0.08)),
    UniformPrior("age_stratification.prop_hospital.multiplier", (0.01, 0.08)),
]

calibration = Calibration(priors=priors, targets=targets, random_process=None, metropolis_init="current_params")

plot_spec_filepath = build_rel_path("timeseries.json")
with open(plot_spec_filepath) as f:
    plot_spec = json.load(f)


def custom_build_model(param_set, build_options=None):
    model = build_model(param_set, build_options)
    for v in roc_vars:
        model.request_function_output(
            v + '_roc',
            wrap_series_transform_for_ndarray(get_roc),
            [v],
        )
    return model


# Create and register the project
project = Project(Region.BANGLADESH, Models.SM_SIR, custom_build_model, param_set, calibration, plots=plot_spec)
