from autumn.tools.project import Project, ParameterSet, TimeSeriesSet, build_rel_path
from autumn.tools.calibration import Calibration, priors, targets
from autumn.tools.calibration.priors import UniformPrior
from autumn.tools.calibration.targets import NormalTarget
from autumn.models.covid_19 import base_params, build_model
from autumn.settings import Region, Models

# Load and configure model parameters
vic_base_path = build_rel_path("../vic_submodel_params.yml")
baseline_params = base_params.update(vic_base_path).update(build_rel_path("params/default.yml"))
param_set = ParameterSet(baseline=baseline_params, scenarios=[])

# Load and configure calibration settings.
ts_set = TimeSeriesSet.from_file(build_rel_path("timeseries.json"))
priors = [
    UniformPrior("contact_rate", [0.025, 0.05]),
]
targets = [
    NormalTarget(
        timeseries=ts_set["notifications"],
        time_weights=list(range(1, len(ts_set["notifications"].times) + 1)),
    )
]
calibration = Calibration(priors=priors, targets=targets)
project = Project(Region.HUME, Models.COVID_19, build_model, param_set, calibration)
