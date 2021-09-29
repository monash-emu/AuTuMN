from autumn.tools.project import Project, ParameterSet, TimeSeriesSet, build_rel_path
from autumn.tools.calibration import Calibration, priors, targets
from autumn.tools.calibration.priors import UniformPrior
from autumn.tools.calibration.targets import NormalTarget
from autumn.models.covid_19 import base_params, build_model
from autumn.settings import Region, Models


# Load and configure model parameters.
baseline_params = base_params.update(build_rel_path("params/default.yml"))
param_set = ParameterSet(baseline=baseline_params, scenarios=[])


# Load and configure calibration settings.
ts_set = TimeSeriesSet.from_file(build_rel_path("timeseries.json"))
priors = [
    UniformPrior("contact_rate", [0.025, 0.05]),
    UniformPrior("recovery_rate", [0.9, 1.2]),
]
targets = [
    NormalTarget(
        timeseries=ts_set["prevalence_infectious"],
        time_weights=list(range(1, len(ts_set["prevalence_infectious"].times) + 1)),
    )
]
calibration = Calibration(priors=priors, targets=targets)
project = Project(Region.BARWON_SOUTH_WEST, Models.EXAMPLE, build_model, param_set, calibration)
