from autumn.tools.project import Project, ParameterSet, load_timeseries, build_rel_path
from autumn.runners.calibration import Calibration
from autumn.runners.calibration.priors import UniformPrior, BetaPrior
from autumn.runners.calibration.targets import (
    NormalTarget,
    get_dispersion_priors_for_gaussian_targets,
)
from autumn.models.covid_19 import base_params, build_model
from autumn.settings import Region, Models

from autumn.projects.covid_19.calibration import COVID_GLOBAL_PRIORS


# Load and configure model parameters.
malaysia_path = build_rel_path("../malaysia/params/default.yml")
default_path = build_rel_path("params/default.yml")
scenario_paths = [build_rel_path(f"params/scenario-{i}.yml") for i in range(10, 12)]
mle_path = build_rel_path("params/mle-params.yml")
baseline_params = (
    base_params.update(malaysia_path).update(default_path).update(mle_path, calibration_format=True)
)
scenario_params = [baseline_params.update(p) for p in scenario_paths]
param_set = ParameterSet(baseline=baseline_params, scenarios=scenario_params)

ts_set = load_timeseries(build_rel_path("timeseries.json"))
notifications_ts = ts_set["notifications"].loc[270:]
targets = [NormalTarget(notifications_ts)]
priors = [
    # Global COVID priors
    *COVID_GLOBAL_PRIORS,
    # Dispersion parameters based on targets
    *get_dispersion_priors_for_gaussian_targets(targets),
    # Regional parameters
    UniformPrior("contact_rate", [0.015, 0.06]),
    UniformPrior("infectious_seed", [30.0, 200.0]),
    # Detection
    UniformPrior("testing_to_detection.assumed_cdr_parameter", [0.03, 0.15]),
    # Microdistancing
    UniformPrior("mobility.microdistancing.behaviour.parameters.end_asymptote", [0.1, 0.4]),
    # Health system-related
    UniformPrior("clinical_stratification.props.hospital.multiplier", [0.7, 1.3]),
    UniformPrior("clinical_stratification.icu_prop", [0.12, 0.25]),
    UniformPrior("clinical_stratification.non_sympt_infect_multiplier", [0.15, 0.4]),
    UniformPrior("clinical_stratification.props.symptomatic.multiplier", [0.8, 2.0]),
    UniformPrior("vaccination.coverage_override", [0.0, 1.0], sampling="lhs"),
    BetaPrior("vaccination.one_dose.ve_prop_prevent_infection", mean=0.7, ci=[0.5, 0.9], sampling="lhs"),
    UniformPrior("vaccination.one_dose.ve_sympt_covid", [0.0, 1.0], sampling="lhs"),
    UniformPrior("voc_emergence.alpha_beta.contact_rate_multiplier", [1.0, 3.0]),
    UniformPrior("voc_emergence.delta.contact_rate_multiplier", [2.0, 5.0]),
    UniformPrior("voc_emergence.alpha_beta.start_time", [275, 450]),
    UniformPrior("voc_emergence.delta.start_time", [450, 600]),
]

calibration = Calibration(priors, targets)

# FIXME: Replace with flexible Python plot request API.
import json

plot_spec_filepath = build_rel_path("timeseries.json")
with open(plot_spec_filepath) as f:
    plot_spec = json.load(f)


project = Project(
    Region.SELANGOR, Models.COVID_19, build_model, param_set, calibration, plots=plot_spec
)
