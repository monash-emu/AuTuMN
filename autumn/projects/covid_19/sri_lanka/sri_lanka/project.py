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
notifications_ts = ts_set.get("notifications").truncate_start_time(350).moving_average(window=7).downsample(step=7)
death_ts = ts_set.get("infection_deaths").truncate_start_time(350)
targets = [
    NormalTarget(notifications_ts),
    NormalTarget(death_ts),
]

priors = [
    # Global COVID priors
    *COVID_GLOBAL_PRIORS,
    # Regional parameters
    UniformPrior("contact_rate", [0.04, 0.06]),
    # Detection
    UniformPrior("infection_fatality.multiplier", [0.17, 1.2]),
    UniformPrior("testing_to_detection.assumed_cdr_parameter", [0.009, 0.025]),
    TruncNormalPrior("clinical_stratification.props.symptomatic.multiplier", mean=1.0, \
                     stdev=0.5, trunc_range=[0.0, np.inf]),
    UniformPrior("contact_tracing.assumed_trace_prop", [0.55, 0.85]),
    #VoC
    UniformPrior("voc_emergence.alpha_beta.start_time", [380, 430]),
    UniformPrior("voc_emergence.delta.start_time", [490, 530]),
    UniformPrior("voc_emergence.alpha_beta.contact_rate_multiplier", [1.0, 3.3]),
    UniformPrior("voc_emergence.delta.contact_rate_multiplier", [1.0, 8.5]),
    TruncNormalPrior(
        "voc_emergence.delta.ifr_multiplier",
        mean=2., stdev=0.75, trunc_range=(1., 4)),
    #waning
    TruncNormalPrior(
        "history.waned.ve_death",
        mean=.5, stdev=.25, trunc_range=(0.28, 1)),
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