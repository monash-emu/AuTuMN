import numpy as np
from autumn.tools.calibration.proposal_tuning import perform_all_params_proposal_tuning
from autumn.tools.project import Project, ParameterSet, TimeSeriesSet, build_rel_path, get_all_available_scenario_paths, \
    use_tuned_proposal_sds
from autumn.tools.calibration import Calibration
from autumn.tools.calibration.priors import UniformPrior, BetaPrior,TruncNormalPrior
from autumn.tools.calibration.targets import (
    NormalTarget,
    get_dispersion_priors_for_gaussian_targets,
)
from autumn.models.covid_19 import base_params, build_model
from autumn.settings import Region, Models

from autumn.projects.covid_19.calibration import COVID_GLOBAL_PRIORS
from autumn.projects.covid_19.sri_lanka.sri_lanka.scenario_builder import get_all_scenario_dicts

# Load and configure model parameters.
default_path = build_rel_path("params/default.yml")
#scenario_paths = [build_rel_path(f"params/scenario-{i}.yml") for i in range(1, 2)]
mle_path = build_rel_path("params/mle-params.yml")
baseline_params = base_params.update(default_path).update(mle_path, calibration_format=True)
all_scenario_dicts = get_all_scenario_dicts("LKA")
#scenario_params = [baseline_params.update(p) for p in scenario_paths]
scenario_params = [baseline_params.update(sc_dict) for sc_dict in all_scenario_dicts]
param_set = ParameterSet(baseline=baseline_params, scenarios=scenario_params)

ts_set = TimeSeriesSet.from_file(build_rel_path("timeseries.json"))
notifications_ts = ts_set.get("notifications").truncate_start_time(350)
death_ts = ts_set.get("infection_deaths").truncate_start_time(350)
targets = [
    NormalTarget(notifications_ts),
    NormalTarget(death_ts),
]

priors = [
    # Global COVID priors
    *COVID_GLOBAL_PRIORS,
    # Regional parameters
    UniformPrior("contact_rate", [0.035, 0.075]),
    UniformPrior("infectious_seed", [450, 500]),
    # Detection
    UniformPrior("infection_fatality.multiplier", [0.1, 1.25]),
    UniformPrior("testing_to_detection.assumed_cdr_parameter", [0.001, 0.02]),
    UniformPrior("contact_tracing.assumed_trace_prop", [0.55, 0.9]),
    UniformPrior("hospital_reporting", [0.25, 0.75]),
    TruncNormalPrior("clinical_stratification.props.symptomatic.multiplier",
                mean =1.0,
                stdev=0.3,
                trunc_range=[0.3, np.inf],
    ),
    #VoC
    UniformPrior("voc_emergence.alpha_beta.start_time", [400, 470]),
    UniformPrior("voc_emergence.delta.start_time", [515, 550]),
    UniformPrior("voc_emergence.alpha_beta.contact_rate_multiplier", [0.5, 5.0]),
    UniformPrior("voc_emergence.delta.contact_rate_multiplier", [0.5, 10.0]),
    #waning
    UniformPrior("history.natural_immunity_duration", [365, 475])
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