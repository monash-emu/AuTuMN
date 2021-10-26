import json

from autumn.tools.project import Project, ParameterSet, TimeSeriesSet, build_rel_path
from autumn.tools.calibration import Calibration
from autumn.models.covid_19 import base_params, build_model
from autumn.settings import Region, Models
from autumn.projects.covid_19.victoria.region_calibration import priors, collate_regional_targets
from autumn.tools.calibration.priors import UniformPrior

# Load and configure model parameters
vic_base_path = build_rel_path("../vic_submodel_params.yml")
vic_regional_path = build_rel_path("../vic_regional_submodel_params.yml")
cluster_path = build_rel_path("./params/default.yml")
baseline_params = base_params.update(vic_base_path).update(vic_regional_path).update(cluster_path)
scenario_path = build_rel_path("../regional_roadmap.yml")
scenario_params = [baseline_params.update(scenario_path)]
param_set = ParameterSet(baseline=baseline_params, scenarios=scenario_params)

# Load and configure calibration settings
ts_set = TimeSeriesSet.from_file(build_rel_path("targets.secret.json"))
targets = collate_regional_targets(ts_set)
priors[0] = UniformPrior("contact_rate", (0.1, 0.24), jumping_stdev=0.05)  # Needs to go a bit higher for some reason
calibration = Calibration(priors=priors, targets=targets)
plot_spec_filepath = build_rel_path("targets.secret.json")
with open(plot_spec_filepath) as f:
    plot_spec = json.load(f)
project = Project(Region.BARWON_SOUTH_WEST, Models.COVID_19, build_model, param_set, calibration, plots=plot_spec)
