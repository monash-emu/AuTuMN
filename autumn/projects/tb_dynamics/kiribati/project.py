from autumn.core.project import (
    Project,
    ParameterSet,
    load_timeseries,
    build_rel_path,
    use_tuned_proposal_sds,
)
from autumn.calibration import Calibration
from autumn.calibration.priors import UniformPrior
from autumn.calibration.targets import (
    NormalTarget,
    get_dispersion_priors_for_gaussian_targets,
)
from autumn.models.tuberculosis import base_params, build_model
from autumn.settings import Region, Models

# Load and configure model parameters.
default_path = build_rel_path("params/default.yml")
baseline_params = base_params.update(default_path)


param_set = ParameterSet(baseline=baseline_params)


project = Project(
    Region.KIRIBATI,
    Models.TBD,
    build_model,
    param_set,
)
