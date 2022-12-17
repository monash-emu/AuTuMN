from autumn.core.project import (
    Project,
    ParameterSet,
    load_timeseries,
    build_rel_path,
)
from autumn.calibration import Calibration
from autumn.calibration.priors import UniformPrior
from autumn.calibration.targets import NormalTarget
from autumn.models.tb_dynamics import base_params, build_model
from autumn.settings import Region, Models
from autumn.projects.tb_dynamics.vietnam.ca_mau.calibration_utils import get_natural_history_priors_from_cid

# Load and configure model parameters.
default_path = build_rel_path("params/default.yml")
mle_path = build_rel_path("params/mle-params.yml")
baseline_params = base_params.update(default_path).update(mle_path, calibration_format=True)
ts_set = load_timeseries(build_rel_path("timeseries.json"))
param_set = ParameterSet(baseline=baseline_params, scenarios=[])

targets = [
    NormalTarget(ts_set["total_population"], stdev=2500.0),
]
# Add uncertainty around natural history using our CID estimates
natural_history_priors = []
for param_name in ["infect_death_rate", "self_recovery_rate"]:
    for organ in ["smear_positive", "smear_negative"]:
        prior = get_natural_history_priors_from_cid(param_name, organ)
        natural_history_priors.append(prior)

priors = [
    UniformPrior("start_population_size", [180000, 300000]),
    UniformPrior("contact_rate", [0.002, 0.01]),
    UniformPrior("infectious_seed", [100, 2000]),
    UniformPrior("rr_infection_latent", [0.2, 0.5]),
    UniformPrior("rr_infection_recovered", [0.2, 1.0]),
    UniformPrior("time_variant_tb_screening_rate.inflection_time", [1990.0, 2020.0]),
    UniformPrior("time_variant_tb_screening_rate.shape", [0.07, 0.1]),
    UniformPrior("time_variant_tb_screening_rate.end_asymptote", [0.4, 0.55]),
    UniformPrior("progression_multiplier", [0.5, 2.0]),
    *natural_history_priors
]
calibration = Calibration(
    priors, targets, metropolis_init="current_params", metropolis_init_rel_step_size=0.1
)

project = Project(
    Region.CA_MAU,
    Models.TBD,
    build_model,
    param_set,
    calibration,
)
