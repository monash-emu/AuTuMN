from email.mime import base
from autumn.tools.project import Project, ParameterSet, load_timeseries, build_rel_path
from autumn.tools.calibration import Calibration
from autumn.tools.calibration.priors import UniformPrior, HierarchicalPrior
from autumn.tools.calibration.targets import NormalTarget
from autumn.models.hierarchical_sir import base_params, build_model
from autumn.settings import Region, Models

# Load and configure model parameters

baseline_params = base_params.update(build_rel_path("params/baseline.yml"))
param_set = ParameterSet(baseline=baseline_params)

# Load and configure calibration settings
ts_set = load_timeseries(build_rel_path("timeseries.json"))
priors = [
    UniformPrior("hyper_beta", [0., 1.]),
    UniformPrior("gamma", [0.9, 1.2]),
]

hierarchical_priors = [
    HierarchicalPrior("beta.AUS", "normal", ['hyper_beta', .01]),
    HierarchicalPrior("beta.ITA", "normal", ['hyper_beta', .01]),
]

targets = [
    NormalTarget(data=ts_set["incidence_AUS"].loc[61:153]),
    NormalTarget(data=ts_set["incidence_ITA"].loc[61:153]),
]
calibration = Calibration(priors=priors, targets=targets, hierarchical_priors=hierarchical_priors)

# Create and register the project
project = Project(Region.MULTI, Models.HIERARCHICAL_SIR, build_model, param_set, calibration)
