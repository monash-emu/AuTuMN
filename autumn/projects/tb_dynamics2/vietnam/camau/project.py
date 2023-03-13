from autumn.core.project import (
    Project,
    ParameterSet,
    load_timeseries,
    build_rel_path,
)
from autumn.calibration import Calibration
from autumn.calibration.priors import UniformPrior
from autumn.calibration.targets import NormalTarget
from autumn.models.tb_dynamics2 import get_base_params, build_model

from autumn.settings import Region, Models
import json

# Load and configure model parameters.
base_params = get_base_params()
default_path = build_rel_path("params/default.yml")
baseline_params = base_params.update(default_path)
ts_set = load_timeseries(build_rel_path("timeseries.json"))
param_set = ParameterSet(baseline=baseline_params, scenarios=[])

targets = [
    NormalTarget(ts_set["total_population"], stdev=2500.0),
    NormalTarget(ts_set["notifications"], stdev=20.0)
]

priors = [
    UniformPrior("start_population_size", [100000, 400000]),
    UniformPrior("contact_rate", [0.0001, 0.02]),
    UniformPrior("rr_infection_latent", [0.2, 0.5]),
    UniformPrior("rr_infection_recovered", [0.1, 0.5]),
    UniformPrior("progression_multiplier", [0.5, 2.0]),
    UniformPrior("cdr_adjustment", [0.6, 1.0]),
    UniformPrior("infect_death_rate_dict.smear_positive", [0.335, 0.449]),
    UniformPrior("infect_death_rate_dict.smear_negative", [0.017, 0.035]),
    UniformPrior("self_recovery_rate_dict.smear_positive", [0.177, 0.288]),
    UniformPrior("self_recovery_rate_dict.smear_negative", [0.073, 0.209]),
    #UniformPrior("gender.adjustments.infection.male", [1.0, 4.0])
]
calibration = Calibration(
    priors=priors,
    targets=targets,
    metropolis_init="current_params",  # "lhs"
    haario_scaling_factor=1.2, # 2.4,
    fixed_proposal_steps=500,
    metropolis_init_rel_step_size=0.1,
    using_summer2=True,
)

plot_spec_filepath = build_rel_path("timeseries.json")
with open(plot_spec_filepath) as f:
    plot_spec = json.load(f)
plot_spec

project = Project(
    Region.CAMAU,
    Models.TBD2,
    build_model,
    param_set,
    calibration,
    plots=plot_spec
)
