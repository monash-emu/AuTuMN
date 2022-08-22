import datetime
import json
from email.mime import base
from typing import Dict

import pandas as pd
from autumn.calibration import Calibration
from autumn.calibration.priors import UniformPrior
from autumn.calibration.targets import NormalTarget
from autumn.core.project import (
    ParameterSet, Project, build_rel_path, load_timeseries, get_all_available_scenario_paths
)
from autumn.models.sm_sir import base_params, build_model
from autumn.settings import Models, Region
from summer.utils import ref_times_to_dti


def get_ts_date_indexes(
    ts_set: Dict[str, pd.Series],
    ref_time: datetime.datetime,
) -> Dict[str, pd.Series]:
    """
    Get a version of a time series set, but with each
    constituent time series having date indexes,
    rather than integer.

    Args:
        ts_set: The time series set we want to change
        ref_time: The model's reference time
    Returns:
        A modified version of the time series set

    """
    new_ts_set = {}
    for indicator, ts in ts_set.items():
        new_ts = ts.copy()
        new_ts.index = ref_times_to_dti(ref_time, new_ts.index)
        new_ts_set[indicator] = new_ts
    return new_ts_set


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

ts_path = build_rel_path("timeseries.secret.json")

# Load and configure calibration settings
ts_set = load_timeseries(ts_path)
priors = [
    UniformPrior("contact_rate", (0.03, 0.08)),
    UniformPrior("voc_emergence.ba_1.cross_protection.ba_2.early_reinfection", (0.1, 0.5)),
    UniformPrior("detect_prop", (0.25, 0.5)),
    UniformPrior("sojourns.active", (2., 4.)),
]
start_time = baseline_params["time"]["start"]
end_emergency_period = 895  # 13th June 2022
hosp_first_above_20 = 744  # 13th January 2022
hosp_last_above_20 = 780  # 18th February 2022
targets = [
    NormalTarget(data=ts_set["notifications"].loc[start_time: end_emergency_period]),
    NormalTarget(data=ts_set["hospital_admissions"].loc[hosp_first_above_20: hosp_last_above_20]),
]

calibration = Calibration(
    priors=priors, targets=targets, random_process=None, metropolis_init="current_params"
)

with open(ts_path) as f:
    plot_spec = json.load(f)

# Create and register the project
project = Project(
    Region.NORTHERN_TERRITORY,
    Models.SM_SIR,
    build_model,
    param_set,
    calibration,
    plots=plot_spec,
    ts_set=ts_set,
)
