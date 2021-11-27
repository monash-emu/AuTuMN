import json

import numpy as np

from autumn.tools.project import Project, ParameterSet, TimeSeriesSet, build_rel_path, get_all_available_scenario_paths
from autumn.tools.calibration import Calibration
from autumn.tools.calibration.priors import UniformPrior, TruncNormalPrior
from autumn.tools.calibration.targets import NormalTarget
from autumn.models.covid_19 import base_params, build_model
from autumn.settings import Region, Models


# Load and configure model parameters
mle_path = build_rel_path("params/mle-params.yml")
default_path = build_rel_path("params/default.yml")
baseline_params = base_params.update(default_path).update(mle_path, calibration_format=True)
scenario_dir_path = build_rel_path("params/")
scenario_paths = get_all_available_scenario_paths(scenario_dir_path)
scenario_params = [baseline_params.update(p) for p in scenario_paths]
param_set = ParameterSet(baseline=baseline_params, scenarios=scenario_params)

# Load and configure calibration settings
ts_set = TimeSeriesSet.from_file(build_rel_path("timeseries.json"))
notifications = ts_set.get("notifications").multiple_truncations([[199, 415], [561, 730]])
infection_deaths = ts_set.get("infection_deaths").truncate_start_time(199)
targets = [
    NormalTarget(notifications),
    # NormalTarget(infection_deaths)
]

priors = [
    TruncNormalPrior(
        "sojourn.compartment_periods_calculated.exposed.total_period",
        mean=5.5, stdev=0.7, trunc_range=(1.0, np.inf)),
    TruncNormalPrior(
        "sojourn.compartment_periods_calculated.active.total_period",
        mean=6.5, stdev=0.77, trunc_range=(4.0, np.inf)),
    TruncNormalPrior(
        "waning_immunity_duration",
        mean=365., stdev=90., trunc_range=(180., np.inf)),
    TruncNormalPrior(
        "vaccination.vacc_part_effect_duration",
        mean=365., stdev=90., trunc_range=(180., np.inf)),
    TruncNormalPrior(
        "voc_emergence.delta.contact_rate_multiplier",
        mean=2.0, stdev=0.1, trunc_range=(1.0, np.inf),
    ),
    UniformPrior("voc_emergence.delta.start_time", (340., 375.), jumping_stdev=8.),  # Tighten up the lower limit to avoid wild runs
    UniformPrior("contact_rate", (0.05, 0.08), jumping_stdev=0.01),  # Tighten up the lower limit to avoid wild runs
    UniformPrior("infectious_seed", (50., 500.), jumping_stdev=40.),
    UniformPrior("testing_to_detection.assumed_cdr_parameter", (0.004, 0.012), jumping_stdev=0.002),
    UniformPrior("mobility.microdistancing.behaviour.parameters.end_asymptote", (0.1, 0.3), jumping_stdev=0.05),
]
calibration = Calibration(priors=priors, targets=targets)

plot_spec_filepath = build_rel_path("timeseries.json")
with open(plot_spec_filepath) as f:
    plot_spec = json.load(f)

project = Project(Region.MYANMAR, Models.COVID_19, build_model, param_set, calibration, plots=plot_spec)
