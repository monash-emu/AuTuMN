from autumn.tools.project import Project, ParameterSet, TimeSeriesSet, build_rel_path
from autumn.tools.calibration import Calibration, priors, targets
from autumn.tools.calibration.priors import UniformPrior
from autumn.tools.calibration.targets import NormalTarget
from autumn.models.example import base_params, build_model
from autumn.settings import Region, Models


# Load and configure model parameters.
default_params = base_params.update(build_rel_path("params/default.yml"))
scenario_1_params = default_params.update(build_rel_path("params/scenario-1.yml"))
param_set = ParameterSet(baseline=default_params, scenarios=[scenario_1_params])


# Load and configure calibration settings.
ts_set = TimeSeriesSet.from_file(build_rel_path("timeseries.json"))
priors = [
    UniformPrior("contact_rate", [0.025, 0.05]),
    UniformPrior("recovery_rate", [0.9, 1.2]),
]
targets = [
    NormalTarget(
        timeseries=ts_set["notifications"],
        time_weights=list(range(1, len(ts_set["notifications"].times) + 1)),
    )
]
calibration = Calibration(priors=priors, targets=targets)
project = Project(Region.MYANMAR, Models.COVID_19, build_model, param_set, calibration)
