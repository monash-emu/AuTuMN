import json
import numpy as np

from autumn.tools.project import Project, ParameterSet, load_timeseries, build_rel_path
from autumn.runners.calibration import Calibration
from autumn.models.covid_19 import base_params, build_model
from autumn.settings import Region, Models
from autumn.projects.covid_19.victoria.region_calibration import priors, collate_metro_targets
from autumn.runners.calibration.priors import TruncNormalPrior

# Load and configure model parameters
vic_base_path = build_rel_path("../vic_submodel_params.yml")
cluster_path = build_rel_path("./params/default.yml")
baseline_params = base_params.update(vic_base_path).update(cluster_path)
scenario_path = build_rel_path("../roadmap.yml")
scenario_params = [baseline_params.update(scenario_path)]
param_set = ParameterSet(baseline=baseline_params, scenarios=scenario_params)

# Load and configure calibration settings
ts_set = load_timeseries(build_rel_path("targets.secret.json"))
targets = collate_metro_targets(ts_set)
priors.append(
    TruncNormalPrior(
        "clinical_stratification.props.hospital.multiplier",
        mean=1., stdev=0.25, trunc_range=(0.5, np.inf), jumping_stdev=0.05
    )
)
calibration = Calibration(priors=priors, targets=targets)
plot_spec_filepath = build_rel_path("targets.secret.json")
with open(plot_spec_filepath) as f:
    plot_spec = json.load(f)
project = Project(Region.SOUTH_EAST_METRO, Models.COVID_19, build_model, param_set, calibration, plots=plot_spec)
