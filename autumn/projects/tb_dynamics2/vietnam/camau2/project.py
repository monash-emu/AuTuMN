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
from pathlib import Path

# Load and configure model parameters.
base_params = get_base_params()
default_path = build_rel_path("params/default.yml")
baseline_params = base_params.update(default_path)
ts_set = load_timeseries(build_rel_path("timeseries.json"))
param_set = ParameterSet(baseline=baseline_params, scenarios=[])

targets = [
    NormalTarget(ts_set["total_population"], stdev=2500.0),
    #NormalTarget(ts_set["notifications"], stdev=40.0)
]


priors = [
    UniformPrior("start_population_size", [180000, 300000]),
    UniformPrior("contact_rate", [0.002, 0.01]),
    UniformPrior("infectious_seed", [100, 2000]),
    UniformPrior("rr_infection_latent", [0.2, 0.5]),
    UniformPrior("rr_infection_recovered", [0.2, 1.0]),
    # UniformPrior("time_variant_tb_screening_rate.inflection_time", [1990.0, 2020.0]),
    # UniformPrior("time_variant_tb_screening_rate.shape", [0.07, 0.1]),
    # UniformPrior("time_variant_tb_screening_rate.end_asymptote", [0.4, 0.55]),
    UniformPrior("progression_multiplier", [0.5, 2.0]),
    # *natural_history_priors
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

project = Project(
    Region.CAMAU,
    Models.TBD2,
    build_model,
    param_set,
    calibration,
)