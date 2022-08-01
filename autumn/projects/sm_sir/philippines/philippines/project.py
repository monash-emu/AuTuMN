import json
import os
from autumn.core.project import Project, ParameterSet, build_rel_path,\
    get_all_available_scenario_paths
from autumn.calibration import Calibration

from autumn.models.sm_sir import base_params, build_model
from autumn.settings import Region, Models
from autumn.projects.sm_sir.common import get_WPRO_priors, get_tartgets


# Load and configure model parameters.
mle_path = build_rel_path("params/mle-params.yml")

# Check whether the specified path exists or not
isExist = os.path.exists(mle_path)

if isExist:
    baseline_params = base_params.update(build_rel_path("params/baseline.yml")).update(
     mle_path, calibration_format=True
 )
else:
    baseline_params = base_params

param_set = ParameterSet(baseline=baseline_params)

# # Load and configure calibration settings
calibration_start_time = param_set.baseline.to_dict()["time"]["start"]

priors = get_WPRO_priors()


directory_path = os.getcwd()
folder_name = os.path.basename(directory_path)

country = f"{folder_name}"
region = f"{folder_name}"

targets = get_tartgets(calibration_start_time, country, region)

calibration = Calibration(
    priors=priors, targets=targets, random_process=None, metropolis_init="current_params"
)

plot_spec_filepath = build_rel_path("timeseries.json")

with open(plot_spec_filepath) as f:
    plot_spec = json.load(f)

## Create and register the project
project = Project(
    country, Models.SM_SIR, build_model, param_set, calibration, plots=plot_spec)







