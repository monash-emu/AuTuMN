from email.mime import base
from autumn.core.project import Project, ParameterSet, load_timeseries, build_rel_path
from autumn.calibration import Calibration
from autumn.calibration.priors import UniformPrior, HierarchicalPrior
from autumn.calibration.targets import NormalTarget
from autumn.models.hierarchical_sir import base_params, build_model
from autumn.settings import Region, Models
import numpy as np

# Load and configure model parameters

baseline_params = base_params.update(build_rel_path("params/baseline.yml"))
param_set = ParameterSet(baseline=baseline_params)

# Load and configure calibration settings
ts_set = load_timeseries(build_rel_path("timeseries.json"))
priors = [
    UniformPrior("hyper_beta_mean", [0., 1.]),
    UniformPrior("hyper_beta_sd", [0., 1.]),
    UniformPrior("gamma", [0.05, 0.2]),
]

hierarchical_priors = [
    HierarchicalPrior("beta.AUS", "trunc_normal", ['hyper_beta_mean', "hyper_beta_sd"], trunc_range=[0., np.inf]),
    HierarchicalPrior("beta.ITA", "trunc_normal", ['hyper_beta_mean', "hyper_beta_sd"], trunc_range=[0., np.inf]),
]

targets = [
    NormalTarget(data=ts_set["incidence_AUS"]),
    NormalTarget(data=ts_set["incidence_ITA"]),
]
calibration = Calibration(priors=priors, targets=targets, hierarchical_priors=hierarchical_priors, metropolis_init='lhs')


import json

plot_spec_filepath = build_rel_path("timeseries.json")
with open(plot_spec_filepath) as f:
    plot_spec = json.load(f)

# Create and register the project
project = Project(Region.MULTI, Models.HIERARCHICAL_SIR, build_model, param_set, calibration, plots=plot_spec)
