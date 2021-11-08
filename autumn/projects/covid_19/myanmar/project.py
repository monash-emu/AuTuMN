import json

from autumn.tools.project import Project, ParameterSet, TimeSeriesSet, build_rel_path
from autumn.tools.calibration import Calibration
from autumn.tools.calibration.priors import UniformPrior
from autumn.tools.calibration.targets import NormalTarget
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
notifications_ts = ts_set.get("notifications").truncate_start_time(500)
infection_deaths_ts = ts_set.get("infection_deaths").truncate_start_time(500)
targets = [
    NormalTarget(notifications_ts),
    # NormalTarget(infection_deaths_ts),
]

priors = [
    *COVID_GLOBAL_PRIORS,
    UniformPrior("contact_rate", (0.02, 0.2), jumping_stdev=0.01),
    UniformPrior("infectious_seed", (50., 500.), jumping_stdev=40.),
    UniformPrior("testing_to_detection.assumed_cdr_parameter", (0.001, 0.01), jumping_stdev=0.002),
    UniformPrior("waning_immunity_duration", (180., 730.), jumping_stdev=90.),
]
calibration = Calibration(priors=priors, targets=targets)

plot_spec_filepath = build_rel_path("timeseries.json")
with open(plot_spec_filepath) as f:
    plot_spec = json.load(f)

project = Project(Region.MYANMAR, Models.COVID_19, build_model, param_set, calibration, plots=plot_spec)
