import json

from autumn.tools.project import Project, ParameterSet, TimeSeriesSet, build_rel_path
from autumn.tools.calibration import Calibration
from autumn.models.covid_19 import base_params, build_model
from autumn.settings import Region, Models
from autumn.projects.covid_19.victoria.region_calibration import priors, collate_metro_targets

# Load and configure model parameters
vic_base_path = build_rel_path("../vic_submodel_params.yml")
cluster_path = build_rel_path("./params/default.yml")
baseline_params = base_params.update(vic_base_path).update(cluster_path)
param_set = ParameterSet(baseline=baseline_params)

# Load and configure calibration settings
ts_set = TimeSeriesSet.from_file(build_rel_path("targets.secret.json"))
targets = collate_metro_targets(ts_set)
calibration = Calibration(priors=priors, targets=targets)
plot_spec_filepath = build_rel_path("targets.secret.json")
with open(plot_spec_filepath) as f:
    plot_spec = json.load(f)
project = Project(Region.WEST_METRO, Models.COVID_19, build_model, param_set, calibration, plots=plot_spec)
