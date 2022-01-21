from autumn.tools.project import Project, ParameterSet, load_timeseries, build_rel_path, get_all_available_scenario_paths
from autumn.tools.calibration import Calibration
from autumn.tools.calibration.priors import UniformPrior, TruncNormalPrior, BetaPrior
from autumn.tools.calibration.targets import (
    NormalTarget,
)
from autumn.models.covid_19 import base_params, build_model
from autumn.settings import Region, Models
import numpy as np
from autumn.projects.covid_19.calibration import COVID_GLOBAL_PRIORS

scenario_dir_path = build_rel_path("params/")

# Load and configure model parameters.
default_path = build_rel_path("params/default.yml")
scenario_paths = get_all_available_scenario_paths(scenario_dir_path)
mle_path = build_rel_path("params/mle-params.yml")
baseline_params = base_params.update(default_path).update(mle_path, calibration_format=True)
scenario_params = [baseline_params.update(p) for p in scenario_paths]
param_set = ParameterSet(baseline=baseline_params, scenarios=scenario_params)

ts_set = load_timeseries(build_rel_path("timeseries.json"))

targets = []
for output_name in ["notifications", "infection_deaths", "icu_occupancy", "hospital_occupancy"]:
    series = ts_set[output_name].loc[491:].rolling(7).mean()  # truncate from May 05th, 2021
    targets.append(NormalTarget(series))

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
    UniformPrior("vaccination.one_dose.ve_prop_prevent_infection", [0, 1], sampling="lhs"),
    BetaPrior("vaccination.one_dose.ve_sympt_covid", mean=0.7, ci=[0.5, 0.9], sampling="lhs"),
]

calibration = Calibration(priors, targets)

# FIXME: Replace with flexible Python plot request API.
import json

plot_spec_filepath = build_rel_path("timeseries.json")
with open(plot_spec_filepath) as f:
    plot_spec = json.load(f)


project = Project(
    Region.HANOI, Models.COVID_19, build_model, param_set, calibration, plots=plot_spec
)
