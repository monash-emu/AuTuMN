import json

from autumn.tools.project import Project, ParameterSet, load_timeseries, build_rel_path,\
    get_all_available_scenario_paths
from autumn.tools.calibration import Calibration
from autumn.tools.calibration.priors import UniformPrior
from autumn.tools.calibration.targets import NormalTarget
from autumn.models.sm_sir import base_params, build_model
from autumn.settings import Region, Models

# Load and configure model parameters.
mle_path = build_rel_path("params/mle-params.yml")
baseline_params = base_params.update(build_rel_path("params/baseline.yml")).update(
    mle_path, calibration_format=True
)
scenario_dir_path = build_rel_path("params/")
scenario_paths = get_all_available_scenario_paths(scenario_dir_path)

scenario_params = [baseline_params.update(p) for p in scenario_paths]
param_set = ParameterSet(baseline=baseline_params, scenarios=scenario_params)

# Load and configure calibration settings
calibration_start_time = param_set.baseline.to_dict()["time"]["start"]
ts_set = load_timeseries(build_rel_path("timeseries.json"))
notifications_end_time = 762
notifications_ts = ts_set["notifications"].loc[calibration_start_time:]
icu_occupancy_ts = ts_set["icu_occupancy"].loc[calibration_start_time:]
hospital_occupancy_ts = ts_set["hospital_occupancy"].loc[calibration_start_time:]
infection_deaths_ts = ts_set["infection_deaths"].loc[calibration_start_time:]
targets = [
    NormalTarget(notifications_ts),
    NormalTarget(icu_occupancy_ts),
    NormalTarget(hospital_occupancy_ts),
    NormalTarget(infection_deaths_ts)
]

priors = [
    UniformPrior("contact_rate", (0.07, 0.13)),
    UniformPrior("testing_to_detection.assumed_cdr_parameter", (0.05, 0.15)),
    UniformPrior("sojourns.latent.total_time", (1.5, 4.0)),
    UniformPrior("voc_emergence.omicron.contact_rate_multiplier", (1.25, 1.5)),
    UniformPrior("voc_emergence.omicron.new_voc_seed.start_time", (520, 675)),
    UniformPrior("voc_emergence.omicron.relative_latency", (0.45, 0.75)),
    UniformPrior("voc_emergence.omicron.relative_active_period", (0.7, 1.3)),
]

calibration = Calibration(
    priors=priors, targets=targets, random_process=None, metropolis_init="current_params"
)

plot_spec_filepath = build_rel_path("timeseries.json")
with open(plot_spec_filepath) as f:
    plot_spec = json.load(f)

# Create and register the project
project = Project(
    Region.MALAYSIA, Models.SM_SIR, build_model, param_set, calibration, plots=plot_spec
)
