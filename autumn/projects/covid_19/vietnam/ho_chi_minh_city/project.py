from autumn.tools.project import Project, ParameterSet, TimeSeriesSet, build_rel_path, get_all_available_scenario_paths
from autumn.tools.calibration import Calibration
from autumn.tools.calibration.priors import UniformPrior, TruncNormalPrior
from autumn.tools.calibration.targets import (
    NormalTarget,
)
from autumn.models.covid_19 import base_params, build_model
from autumn.settings import Region, Models
import numpy as np
from autumn.projects.covid_19.calibration import COVID_GLOBAL_PRIORS


# Load and configure model parameters.
default_path = build_rel_path("params/default.yml")
scenario_dir_path = build_rel_path("params/")
scenario_paths = get_all_available_scenario_paths(scenario_dir_path)
mle_path = build_rel_path("params/mle-params.yml")
baseline_params = base_params.update(default_path).update(mle_path, calibration_format=True)
scenario_params = [baseline_params.update(p) for p in scenario_paths]
param_set = ParameterSet(baseline=baseline_params, scenarios=scenario_params)

ts_set = TimeSeriesSet.from_file(build_rel_path("timeseries.json"))
cutoff_time = 487  # 1 May 2021
notifications_ts = ts_set.get("notifications").truncate_start_time(cutoff_time).moving_average(window=7).downsample(step=7)
hosp_occupancy_ts = ts_set.get("hospital_occupancy").truncate_start_time(cutoff_time).moving_average(window=7).downsample(step=7)
infection_deaths_ts = ts_set.get("infection_deaths").truncate_start_time(cutoff_time).moving_average(window=7).downsample(step=7)
targets = [
    NormalTarget(notifications_ts),
    NormalTarget(hosp_occupancy_ts),
    NormalTarget(infection_deaths_ts),
]

priors = [
    TruncNormalPrior(
        "sojourn.compartment_periods_calculated.exposed.total_period",
        mean=4,
        stdev=0.97,
        trunc_range=[1.0, np.inf],
    ),
    TruncNormalPrior(
        "sojourn.compartment_periods_calculated.active.total_period",
        mean=6.5,
        stdev=0.77,
        trunc_range=[4.0, np.inf],
    ),
    UniformPrior("infectious_seed", [1, 50]),
    UniformPrior("contact_rate", [0.02, 0.03]),
    UniformPrior("infection_fatality.multiplier", [0.5, 3.]),
    UniformPrior("testing_to_detection.assumed_cdr_parameter", [0.001, 0.005]),
]

calibration = Calibration(priors, targets)

# FIXME: Replace with flexible Python plot request API.
import json

plot_spec_filepath = build_rel_path("timeseries.json")
with open(plot_spec_filepath) as f:
    plot_spec = json.load(f)


project = Project(
    Region.HO_CHI_MINH_CITY, Models.COVID_19, build_model, param_set, calibration, plots=plot_spec
)
