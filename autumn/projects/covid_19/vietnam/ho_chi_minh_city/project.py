from autumn.tools.project import Project, ParameterSet, TimeSeriesSet, build_rel_path, get_all_available_scenario_paths
from autumn.tools.calibration import Calibration
from autumn.tools.calibration.priors import UniformPrior, TruncNormalPrior, BetaPrior
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
cutoff_time = 518  # 1 June 2021
hosp_endtime = 582
n_inflated_weight = 35

targets = []
for output_name in ["notifications", "hospital_occupancy", "infection_deaths"]:
    if output_name == "hospital_occupancy":
        series = ts_set.get(output_name).truncate_start_time(cutoff_time).truncate_end_time(hosp_endtime).moving_average(window=7)
    else:
        series = ts_set.get(output_name).truncate_start_time(cutoff_time).moving_average(window=7)

    n = len(series.times)
    max_weight = 10.
    weights = [1.0 for _ in range(n - n_inflated_weight)] + [1.0 + (i + 1) * (max_weight - 1.) / n_inflated_weight for i in range(n_inflated_weight)]

    targets.append(
        NormalTarget(series, time_weights=weights)
    )

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
    UniformPrior("infectious_seed", [1, 20]),
    UniformPrior("contact_rate", [0.035, 0.055]),
    UniformPrior("clinical_stratification.props.hospital.multiplier", [0.5, 3.]),
    UniformPrior("infection_fatality.multiplier", [0.5, 3.]),

    UniformPrior("testing_to_detection.assumed_cdr_parameter", [0.002, 0.007]),
    UniformPrior("mobility.microdistancing.behaviour.parameters.max_effect", [0.1, 0.4]),

    # Vaccination parameters (independent sampling)
    UniformPrior("vaccination.vacc_prop_prevent_infection", [0, 1], sampling="lhs"),
    BetaPrior("vaccination.overall_efficacy", mean=0.7, ci=[0.5, 0.9], sampling="lhs"),
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
