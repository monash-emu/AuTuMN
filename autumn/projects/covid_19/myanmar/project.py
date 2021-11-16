import json

import numpy as np

from autumn.tools.project import Project, ParameterSet, TimeSeriesSet, build_rel_path
from autumn.tools.calibration import Calibration
from autumn.tools.calibration.priors import UniformPrior, TruncNormalPrior
from autumn.tools.calibration.targets import NormalTarget, TruncNormalTarget
from autumn.models.covid_19 import base_params, build_model
from autumn.settings import Region, Models
from autumn.projects.covid_19.calibration import COVID_GLOBAL_PRIORS


# Load and configure model parameters
mle_path = build_rel_path("params/mle-params.yml")
default_path = build_rel_path("params/default.yml")
baseline_params = base_params.update(default_path).update(mle_path, calibration_format=True)

param_set = ParameterSet(baseline=baseline_params, scenarios=[])

# Load and configure calibration settings
ts_set = TimeSeriesSet.from_file(build_rel_path("timeseries.json"))
notifications = ts_set.get("notifications").truncate_start_time(199)
infection_deaths = ts_set.get("infection_deaths").truncate_start_time(199)
targets = [
    NormalTarget(notifications),
    # NormalTarget(infection_deaths)
]

priors = [
    TruncNormalPrior(
        "sojourn.compartment_periods_calculated.exposed.total_period",
        mean=5.5, stdev=0.5, trunc_range=[1.0, np.inf]),
    TruncNormalPrior(
        "sojourn.compartment_periods_calculated.active.total_period",
        mean=6.5, stdev=0.77, trunc_range=[4.0, np.inf]),
    UniformPrior("contact_rate", (0.035, 0.08), jumping_stdev=0.01),
    UniformPrior("infectious_seed", (50., 500.), jumping_stdev=40.),
    UniformPrior("testing_to_detection.assumed_cdr_parameter", (0.004, 0.012), jumping_stdev=0.002),
    UniformPrior("waning_immunity_duration", (180., 730.), jumping_stdev=90.),
    UniformPrior("vaccination.vacc_part_effect_duration", (180., 730.), jumping_stdev=90.),
    UniformPrior("mobility.microdistancing.behaviour.parameters.end_asymptote", (0.1, 0.3), jumping_stdev=0.05),
    UniformPrior("voc_emergence.delta.contact_rate_multiplier", (1.8, 2.2), jumping_stdev=0.1),
]
calibration = Calibration(priors=priors, targets=targets)

plot_spec_filepath = build_rel_path("timeseries.json")
with open(plot_spec_filepath) as f:
    plot_spec = json.load(f)

project = Project(Region.MYANMAR, Models.COVID_19, build_model, param_set, calibration, plots=plot_spec)
