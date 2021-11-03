from autumn.tools.project import Project, ParameterSet, TimeSeriesSet, build_rel_path
from autumn.tools.calibration import Calibration
from autumn.tools.calibration.priors import UniformPrior
from autumn.tools.calibration.targets import NormalTarget
from autumn.models.covid_19 import base_params, build_model
from autumn.settings import Region, Models
from autumn.projects.covid_19.calibration import COVID_GLOBAL_PRIORS


# Load and configure model parameters.
default_params = base_params.update(build_rel_path("params/default.yml"))
param_set = ParameterSet(baseline=default_params, scenarios=[])

# Load and configure calibration settings.
ts_set = TimeSeriesSet.from_file(build_rel_path("timeseries.json"))
notifications_ts = ts_set.get("notifications").truncate_start_time(500)
infection_deaths_ts = ts_set.get("infection_deaths").truncate_start_time(500).downsample(7)
targets = [
    NormalTarget(notifications_ts),
    NormalTarget(infection_deaths_ts),
]

priors = [
    *COVID_GLOBAL_PRIORS,
    UniformPrior("contact_rate", (0.025, 0.05), jumping_stdev=0.008),
    UniformPrior("infectious_seed", (50., 500.), jumping_stdev=40.),
    UniformPrior("testing_to_detection.assumed_cdr_parameter", (0.004, 0.015), jumping_stdev=0.002),
    UniformPrior("time.start", (300., 400.), jumping_stdev=10.),
]
calibration = Calibration(priors=priors, targets=targets)
project = Project(Region.MYANMAR, Models.COVID_19, build_model, param_set, calibration)
