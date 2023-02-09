import json
import os
from autumn.core.project import (
    Project,
    ParameterSet,
    build_rel_path,
    get_all_available_scenario_paths,
)
from autumn.calibration import Calibration
from autumn.calibration.targets import NormalTarget
from autumn.models.sm_sir import base_params, build_model
from autumn.settings import Region, Models
from autumn.projects.sm_sir.WPRO.common import get_WPRO_priors, get_targets, variant_start_time
from autumn.calibration.priors import UniformPrior

# Load and configure model parameters.
mle_path = build_rel_path("params/mle-params.yml")

# Check whether the specified path exists or not
isExistMLE = os.path.exists(mle_path)
baseline_path = build_rel_path("params/baseline.yml")
isExistBaseline = os.path.exists(baseline_path)

if isExistMLE:
    baseline_params = base_params.update(build_rel_path("params/baseline.yml")).update(
        mle_path, calibration_format=True
    )
elif isExistBaseline:
    baseline_params = base_params.update(build_rel_path("params/baseline.yml"))
else:
    baseline_params = base_params

param_set = ParameterSet(baseline=baseline_params)

# # Load and configure calibration settings
calibration_start_time = param_set.baseline.to_dict()["time"]["start"]

variant_times = variant_start_time(["delta", "omicron"], "vietnam")
priors = get_WPRO_priors(variant_times)
priors = priors + [
    UniformPrior("age_stratification.cfr.multiplier", (0.4, 0.8)),
    UniformPrior("contact_rate", (0.07, 0.12)),
    UniformPrior("voc_emergence.omicron.death_protection", (0.5, 1.0)),
]

ts_set = get_targets(calibration_start_time, "vietnam", "vietnam")
infection_deaths_ts = ts_set["infection_deaths"].loc[calibration_start_time:]
notifications_ts = ts_set["notifications"].loc[calibration_start_time:]
targets = [NormalTarget(infection_deaths_ts), NormalTarget(notifications_ts)]

calibration = Calibration(
    priors=priors, targets=targets, random_process=None, metropolis_init="current_params"
)

plot_spec_filepath = build_rel_path("timeseries.json")

with open(plot_spec_filepath) as f:
    plot_spec = json.load(f)

## Create and register the project
project = Project(
    Region.WPRO_VIETNAM, Models.SM_SIR, build_model, param_set, calibration, plots=plot_spec
)
