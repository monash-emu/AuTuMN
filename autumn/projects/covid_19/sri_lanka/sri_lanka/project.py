from autumn.tools.calibration.proposal_tuning import perform_all_params_proposal_tuning
from autumn.tools.project import Project, ParameterSet, TimeSeriesSet, build_rel_path, get_all_available_scenario_paths, \
    use_tuned_proposal_sds
from autumn.tools.calibration import Calibration
from autumn.tools.calibration.priors import UniformPrior, BetaPrior
from autumn.tools.calibration.targets import (
    NormalTarget,
    get_dispersion_priors_for_gaussian_targets,
)
from autumn.models.covid_19 import base_params, build_model
from autumn.settings import Region, Models

from autumn.projects.covid_19.calibration import COVID_GLOBAL_PRIORS


# Load and configure model parameters.
default_path = build_rel_path("params/default.yml")
scenario_paths = [build_rel_path(f"params/scenario-{i}.yml") for i in range(1, 3)]
mle_path = build_rel_path("params/mle-params.yml")
baseline_params = base_params.update(default_path).update(mle_path, calibration_format=True)
scenario_params = [baseline_params.update(p) for p in scenario_paths]
param_set = ParameterSet(baseline=baseline_params, scenarios=scenario_params)

ts_set = TimeSeriesSet.from_file(build_rel_path("timeseries.json"))
notifications_ts = ts_set.get("notifications").truncate_start_time(350).moving_average(window=7).downsample(step=7)
death_ts = ts_set.get("infection_deaths").truncate_start_time(350)
targets = [
    NormalTarget(notifications_ts),
    NormalTarget(death_ts),
]

priors = [
    # Global COVID priors
    *COVID_GLOBAL_PRIORS,
    # Dispersion parameters based on targets
    *get_dispersion_priors_for_gaussian_targets(targets),
    *get_dispersion_priors_for_gaussian_targets(targets),
    # Regional parameters
    UniformPrior("contact_rate", [0.018, 0.023]),
    UniformPrior("infectious_seed", [210.0, 280.0]),
    # Detection
    UniformPrior("testing_to_detection.assumed_cdr_parameter", [0.04, 0.062]),
    UniformPrior("voc_emergence.alpha_beta.start_time", [410, 435]),
    UniformPrior("voc_emergence.alpha_beta.contact_rate_multiplier", [2.75, 3.0]),
    UniformPrior("voc_emergence.delta.start_time", [440, 465]),
    UniformPrior("voc_emergence.delta.contact_rate_multiplier", [3.3, 4.0]),
    UniformPrior("contact_tracing.assumed_trace_prop", [0.1, 0.8]),
    UniformPrior("infection_fatality.multiplier", [2.0, 3.2]),
    # vaccination
    UniformPrior("vaccination.fully_vaccinated.vacc_prop_prevent_infection", [0, 1.0]),
]

# Load proposal sds from yml file
# use_tuned_proposal_sds(priors, build_rel_path("proposal_sds.yml"))

calibration = Calibration(priors, targets)

# FIXME: Replace with flexible Python plot request API.
import json

plot_spec_filepath = build_rel_path("timeseries.json")
with open(plot_spec_filepath) as f:
    plot_spec = json.load(f)


project = Project(
    Region.SRI_LANKA, Models.COVID_19, build_model, param_set, calibration, plots=plot_spec
)

#perform_all_params_proposal_tuning(project, calibration, priors, n_points=50, relative_likelihood_reduction=0.2)
