import pandas as pd

from autumn.tools.project import (
    Project,
    ParameterSet,
    load_timeseries,
    build_rel_path,
    get_all_available_scenario_paths,
    use_tuned_proposal_sds,
)
from autumn.runners.calibration import Calibration
from autumn.runners.calibration.priors import UniformPrior
from autumn.runners.calibration.targets import NormalTarget
from autumn.models.sm_sir import (
    base_params,
    build_model,
    set_up_random_process
)
from autumn.settings import Region, Models

# Load and configure model parameters
mle_path = build_rel_path("params/mle-params.yml")
scenario_dir_path = build_rel_path("params/")
scenario_paths = get_all_available_scenario_paths(scenario_dir_path)

baseline_params = base_params.update(build_rel_path("params/baseline.yml")).update(
    mle_path, calibration_format=True
)
scenario_params = [baseline_params.update(p) for p in scenario_paths]
param_set = ParameterSet(baseline=baseline_params, scenarios=scenario_params)

# Load and configure calibration settings.
ts_set = load_timeseries(build_rel_path("timeseries.json"))

# notifications = pd.concat(
#     [
#      ts_set["notifications"].loc[671:745],  # from 01st Nov 2021 to 14th Jan 2022
#      ts_set["notifications"].loc[763:]  # from 01st Feb 2022 onwards
#     ]
# )

hospital_occupancy = pd.concat(
    [
     ts_set["hospital_occupancy"].loc[671:760],  # from 01st Nov 2021 to 29th Jan 2022
     ts_set["hospital_occupancy"].loc[791:]  # from 1st Mar 2022 onwards
    ]
)

# hospital_occupancy = ts_set["hospital_occupancy"].loc[671:]

infection_deaths = pd.concat(
    [
     ts_set["infection_deaths"].loc[725:763].rolling(7).mean(),  # from 25th Dec 2021 to 01st Feb 2022
     ts_set["infection_deaths"].loc[786:].rolling(7).mean()  # from 24th Feb 2022 onwards
    ]
)

targets = [
    # NormalTarget(notifications),
    NormalTarget(hospital_occupancy),
    NormalTarget(infection_deaths)
]


priors = [
    # infectious seed and contact rate
    # UniformPrior("infectious_seed", (800., 4000.)),
    # UniformPrior("contact_rate", (0.18, 0.25)),
    # detect prop
    # UniformPrior("detect_prop", (0.1, 0.3)),
    # testing to detection params
    # UniformPrior("testing_to_detection.assumed_tests_parameter", (0.0005, 0.002)),
    # UniformPrior("testing_to_detection.assumed_cdr_parameter", (0.01, 0.08)),
    # sojourns
    # UniformPrior("sojourns.latent.total_time", (3, 5.0)),
    # immunity stratification
    # UniformPrior("immunity_stratification.prop_immune", (0.7, 1.0)),
    # UniformPrior("immunity_stratification.prop_high_among_immune", (0.7, 1.0)),
    # age stratification
    UniformPrior("age_stratification.cfr.multiplier", (0.015, 0.055)),
    UniformPrior("age_stratification.prop_hospital.multiplier", (0.02, 0.1)),
    # prop icu among hospitalization
    # UniformPrior("prop_icu_among_hospitalised", (0.01, 0.2)),
    # Omicron-related parameters
    UniformPrior("voc_emergence.omicron.new_voc_seed.start_time", (715.0, 746.0)),  # 3-week interval
    # UniformPrior("voc_emergence.omicron.relative_latency", (0.01, 0.5)),
    UniformPrior("voc_emergence.omicron.contact_rate_multiplier", (1.5, 2.5)),
    # UniformPrior("voc_emergence.omicron.relative_active_period", (0.01, 0.5)),
    # UniformPrior("voc_emergence.omicron.icu_multiplier", (0.1, 0.6)),
    # sojourns
    # UniformPrior("sojourns.active.proportion_early", (0.25, 0.6)),
    # UniformPrior("sojourns.latent.proportion_early", (0., 0.2)),
]


if baseline_params.to_dict()["activate_random_process"]:
    time_params = baseline_params.to_dict()["time"]
    rp = set_up_random_process(time_params["start"], time_params["end"])

    # rp = None  # use this when tuning proposal jumping steps
else:
    rp = None

# Load proposal sds from yml file
use_tuned_proposal_sds(priors, build_rel_path("proposal_sds.yml"))

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
    Region.HANOI, Models.SM_SIR, build_model, param_set, calibration, plots=plot_spec
)


# from autumn.runners.calibration.proposal_tuning import perform_all_params_proposal_tuning
# perform_all_params_proposal_tuning(project, calibration, priors, n_points=20, relative_likelihood_reduction=0.2)
