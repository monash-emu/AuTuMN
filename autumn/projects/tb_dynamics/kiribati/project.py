from autumn.core.project import Project, ParameterSet, load_timeseries, build_rel_path, use_tuned_proposal_sds
from autumn.calibration import Calibration
from autumn.calibration.priors import UniformPrior
from autumn.calibration.targets import (
    NormalTarget,
    get_dispersion_priors_for_gaussian_targets,
)
from autumn.models.tuberculosis import base_params, build_model
from autumn.settings import Region, Models

from autumn.projects.tb_dynamics.kiribati.utils import make_sa_scenario_list

ANALYSIS = "main"
# ANALYSIS = "sa_importation"
# ANALYSIS = "sa_screening"


# Load and configure model parameters.
default_path = build_rel_path("params/default.yml")
baseline_params = base_params.update(default_path)


param_set = ParameterSet(baseline=baseline_params)

# Load and configure calibration targets
ts_set = load_timeseries(build_rel_path("timeseries.json"))
targets = [
    NormalTarget(ts_set["prevalence_infectiousXlocation_starawa"], stdev=80.0),
    NormalTarget(ts_set["percentage_latentXlocation_starawa"], stdev=10.0),
    NormalTarget(ts_set["prevalence_infectiousXlocation_other"], stdev=20.0),
    NormalTarget(ts_set["notificationsXlocation_starawa"], stdev=20.),
    NormalTarget(ts_set["notificationsXlocation_other"], stdev=9.),
    NormalTarget(ts_set["population_size"], stdev=2500.0),
]

# Add uncertainty around natural history using our CID estimates


priors = [
    # *get_dispersion_priors_for_gaussian_targets(targets),
    UniformPrior("start_population_size", [200, 800]),
    UniformPrior("crude_birth_rate", [1, 7]),
]


calibration = Calibration(
    priors, targets, metropolis_init="current_params", metropolis_init_rel_step_size=0.1
)


project = Project(
    Region.KIRIBATI,
    Models.TBD,
    build_model,
    param_set,
    calibration,
)
