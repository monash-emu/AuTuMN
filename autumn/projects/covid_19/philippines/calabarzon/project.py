from autumn.tools.project import Project, ParameterSet, TimeSeriesSet, build_rel_path, get_all_available_scenario_paths, use_tuned_proposal_sds
from autumn.tools.calibration import Calibration

from autumn.models.covid_19 import base_params, build_model
from autumn.settings import Region, Models
import os

from autumn.projects.covid_19.philippines.calibration import get_philippies_calibration_settings


# Load and configure model parameters
phl_base_path = build_rel_path("../phl_submodel_params.yml")
default_path = build_rel_path("params/default.yml")
mle_path = build_rel_path("params/mle-params.yml")
scenario_dir_path = build_rel_path("params/")
scenario_paths = get_all_available_scenario_paths(scenario_dir_path)

baseline_params = (
    base_params.update(phl_base_path).update(default_path).update(mle_path, calibration_format=True)
)
scenario_params = [baseline_params.update(p) for p in scenario_paths]
param_set = ParameterSet(baseline=baseline_params, scenarios=scenario_params)

# Add calibration targets and priors
ts_set = TimeSeriesSet.from_file(build_rel_path("timeseries.json"))
targets, priors = get_philippies_calibration_settings(ts_set)

# Load proposal sds from yml file
use_tuned_proposal_sds(priors, build_rel_path("proposal_sds.yml"))

calibration = Calibration(priors, targets, metropolis_init="current_params")

# FIXME: Replace with flexible Python plot request API.
import json

plot_spec_filepath = build_rel_path("timeseries.json")
with open(plot_spec_filepath) as f:
    plot_spec = json.load(f)

project = Project(
    Region.CALABARZON, Models.COVID_19, build_model, param_set, calibration, plots=plot_spec
)

# from autumn.tools.calibration.proposal_tuning import perform_all_params_proposal_tuning
# perform_all_params_proposal_tuning(project, calibration, priors, n_points=2, relative_likelihood_reduction=0.2)
