import pandas as pd
import numpy as np

from autumn.core.project import (
    Project,
    ParameterSet,
    load_timeseries,
    build_rel_path,
    use_tuned_proposal_sds,
)
from autumn.calibration import Calibration
from autumn.calibration.priors import UniformPrior, TruncNormalPrior
from autumn.calibration.targets import NormalTarget
from autumn.models.sm_sir import base_params, build_model, set_up_random_process
from autumn.settings import Region, Models

# Load and configure model parameters.
mle_path = build_rel_path("params/mle-params.yml")
baseline_params = base_params.update(build_rel_path("params/baseline.yml")).update(
    mle_path, calibration_format=True
)
param_set = ParameterSet(baseline=baseline_params, scenarios=[])

# Load and configure calibration settings.
ts_set = load_timeseries(build_rel_path("timeseries.json"))

notifications = pd.concat(
    [ts_set["notifications"].loc[s] for s in (slice(199, 415), slice(560, 730))]
)
infection_deaths = ts_set["infection_deaths"].loc[199:]
priors = [
    # TruncNormalPrior(
    #     "sojourn.compartment_periods_calculated.exposed.total_period",
    #     mean=5.5,
    #     stdev=0.7,
    #     trunc_range=(1.0, np.inf),
    # ),
    # TruncNormalPrior(
    #     "sojourn.compartment_periods_calculated.active.total_period",
    #     mean=6.5,
    #     stdev=0.77,
    #     trunc_range=(4.0, np.inf),
    # ),
    # TruncNormalPrior(
    #     "history.natural_immunity_duration", mean=365.0, stdev=120.0, trunc_range=(180.0, np.inf)
    # ),
    # TruncNormalPrior(
    #     "vaccination.vacc_part_effect_duration",
    #     mean=365.0,
    #     stdev=120.0,
    #     trunc_range=(180.0, np.inf),
    # ),
    # UniformPrior(
    #     "contact_rate", (0.05, 0.08), jumping_stdev=0.01
    # ),  # Tighten up the lower limit to avoid wild runss
    # UniformPrior("infectious_seed", (50.0, 500.0), jumping_stdev=40.0),
    UniformPrior("testing_to_detection.assumed_cdr_parameter", (0.004, 0.012), jumping_stdev=0.002),
    # UniformPrior(
    #     "mobility.microdistancing.behaviour.parameters.end_asymptote",
    #     (0.1, 0.3),
    #     jumping_stdev=0.05,
    # ),
    # UniformPrior("voc_emergence.delta.contact_rate_multiplier", (1.8, 2.4), jumping_stdev=0.1),
    # UniformPrior("voc_emergence.delta.start_time", (330.0, 390.0), jumping_stdev=30.0),
]

targets = [
    NormalTarget(notifications),
    # NormalTarget(infection_deaths)
]

if baseline_params.to_dict()["activate_random_process"]:
    time_params = baseline_params.to_dict()["time"]
    rp = set_up_random_process(time_params["start"], time_params["end"])

    # rp = None  # use this when tuning proposal jumping steps
else:
    rp = None

# Load proposal sds from yml file
# use_tuned_proposal_sds(priors, build_rel_path("proposal_sds.yml"))

calibration = Calibration(
    priors=priors, targets=targets, random_process=rp, metropolis_init="current_params"
)

# FIXME: Replace with flexible Python plot request API.
import json

plot_spec_filepath = build_rel_path("timeseries.json")
with open(plot_spec_filepath) as f:
    plot_spec = json.load(f)


# Create and register the project.
project = Project(
    Region.MYANMAR, Models.SM_SIR, build_model, param_set, calibration, plots=plot_spec
)


# from autumn.calibration.proposal_tuning import perform_all_params_proposal_tuning
# perform_all_params_proposal_tuning(project, calibration, priors, n_points=50, relative_likelihood_reduction=0.2)
