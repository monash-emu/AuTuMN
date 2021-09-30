from autumn.tools.project import Project, ParameterSet, TimeSeriesSet, build_rel_path
from autumn.tools.calibration import Calibration
from autumn.tools.calibration.targets import NormalTarget
from autumn.models.covid_19 import base_params, build_model
from autumn.settings import Region, Models
from autumn.projects.covid_19.victoria.region_calibration import priors


# Load and configure model parameters
vic_base_path = build_rel_path("../vic_submodel_params.yml")
baseline_params = base_params.update(vic_base_path)
param_set = ParameterSet(baseline=baseline_params)

# Load and configure calibration settings
ts_set = TimeSeriesSet.from_file(build_rel_path("targets.secret.json"))
targets = [NormalTarget(timeseries=ts_set["notifications"])]
calibration = Calibration(priors=priors, targets=targets)
project = Project(Region.SOUTH_METRO, Models.COVID_19, build_model, param_set, calibration)
