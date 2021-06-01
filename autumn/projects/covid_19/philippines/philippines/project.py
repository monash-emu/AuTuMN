from autumn.tools.project import Project, ParameterSet, TimeSeriesSet, build_rel_path
from autumn.tools.calibration import Calibration
from autumn.models.covid_19 import base_params, build_model
from autumn.settings import Region, Models

from autumn.projects.covid_19.philippines.calibration import get_philippies_calibration_settings


# Load and configure model parameters
phl_base_path = build_rel_path("../phl_submodel_params.yml")
default_path = build_rel_path("params/default.yml")
mle_path = build_rel_path("params/mle-params.yml")
scenario_paths = [build_rel_path(f"params/scenario-{i}.yml") for i in range(1, 4)]
baseline_params = (
    base_params.update(phl_base_path).update(default_path).update(mle_path, calibration_format=True)
)
scenario_params = [baseline_params.update(p) for p in scenario_paths]
param_set = ParameterSet(baseline=baseline_params, scenarios=scenario_params)

# Add calibration targets and priors
ts_set = TimeSeriesSet.from_file(build_rel_path("timeseries.json"))
targets, priors = get_philippies_calibration_settings(ts_set)
calibration = Calibration(priors, targets, metropolis_init="current_params")

# FIXME: Replace with flexible Python plot request API.
import json

plot_spec_filepath = build_rel_path("timeseries.json")
with open(plot_spec_filepath) as f:
    plot_spec = json.load(f)

project = Project(
    Region.PHILIPPINES, Models.COVID_19, build_model, param_set, calibration, plots=plot_spec
)
