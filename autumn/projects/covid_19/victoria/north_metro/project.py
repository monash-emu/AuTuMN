from autumn.tools.project import Project, ParameterSet, TimeSeriesSet, build_rel_path
from autumn.tools.calibration import Calibration, priors, targets
from autumn.tools.calibration.priors import UniformPrior
from autumn.tools.calibration.targets import NormalTarget, PoissonTarget
from autumn.models.example import base_params, build_model
from autumn.settings import Region, Models


# Load and configure model parameters.
default_path = build_rel_path("params/default.yml")
mle_path = build_rel_path("params/mle-params.yml")
scenario_paths = [build_rel_path("params/scenario-1.yml")]
baseline_params = base_params.update(default_path)
scenario_1_params = baseline_params.update(build_rel_path("params/scenario-1.yml"))
param_set = ParameterSet(baseline=baseline_params, scenarios=[scenario_1_params])

ts_set = TimeSeriesSet.from_file(build_rel_path("targets.secret.json"))
target_start_time = 600

# Load and configure calibration settings.
ts_set = TimeSeriesSet.from_file(build_rel_path("targets.secret.json"))
priors = [
    UniformPrior("contact_rate", (0.1, 0.28), jumping_stdev=0.008),
    UniformPrior("recovery_rate", [0.9, 1.2]),
]
targets = [
    PoissonTarget(
        ts_set.get("notifications").round_values().truncate_start_time(target_start_time)
    ),
    PoissonTarget(ts_set.get("hospital_admissions").truncate_start_time(target_start_time)),
    PoissonTarget(ts_set.get("icu_admissions").truncate_start_time(target_start_time)),
]
calibration = Calibration(priors=priors, targets=targets)
project = Project(Region.NORTH_METRO, Models.EXAMPLE, build_model, param_set, calibration)
