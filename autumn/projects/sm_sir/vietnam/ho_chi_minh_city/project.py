import pandas as pd

from autumn.core.project import (
    Project,
    ParameterSet,
    load_timeseries,
    build_rel_path,
    get_all_available_scenario_paths,
    use_tuned_proposal_sds,
)
from autumn.calibration import Calibration
from autumn.calibration.priors import UniformPrior
from autumn.calibration.targets import NormalTarget
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

# notifications = ts_set["notifications"].multiple_truncations([[511, 575], [606, 700]])
# truncated from 18th Jul to 28th Jul, then from 28th Aug onwards
# notifications = pd.concat(
#     [
#      ts_set["notifications"].loc[606:639],  # form 28/08/2021 to 30/09/2021
#      ts_set["notifications"].loc[702:]  # from 02/12/2021 onwards
#     ]
# )
hospital_occupancy = pd.concat(
    [
        ts_set["hospital_occupancy"].loc[592:615],  # from 14/08/2021 to 06/06/2021
        ts_set["hospital_occupancy"].loc[632:732],  # truncated from 23 Sep 2021 to 01st Jan 2022
    ]
)

icu_occupancy = ts_set["icu_occupancy"].loc[618:732]  # truncated from 09 Sep 2021 to 01st Jan 2022
# infection_deaths = ts_set["infection_deaths"].loc[556:].rolling(7).mean()  # truncated to 9th Jul 2021

targets = [
    # NormalTarget(notifications),
    NormalTarget(hospital_occupancy),
    NormalTarget(icu_occupancy),
    # NormalTarget(infection_deaths)
]


priors = [
    # age stratification
    # UniformPrior("age_stratification.cfr.multiplier", (0.4, 1.0)),
    # UniformPrior("age_stratification.prop_hospital.multiplier", (0.5, 1.0)),
    # infectious seed and contact rate
    UniformPrior("infectious_seed", (200, 4000)),
    UniformPrior("contact_rate", (0.05, 0.2)),
    # testing to detection params
    # UniformPrior("testing_to_detection.assumed_tests_parameter", (0.001, 0.02)),
    # UniformPrior("testing_to_detection.assumed_cdr_parameter", (0.01, 0.1)),
    # sojourns
    # UniformPrior("sojourns.latent.total_time", (3, 5.0)),
    # immunity stratification
    # UniformPrior("immunity_stratification.prop_immune", (0.7, 0.9)),
    # UniformPrior("immunity_stratification.prop_high_among_immune", (0.0, 1.0)),
    # Microdistancing
    UniformPrior("mobility.microdistancing.behavior.parameters.max_effect", (0.1, 0.5)),
    # prop icu among hospitalization
    # UniformPrior("prop_icu_among_hospitalised", (0.03, 0.1)),
    # emergence of omicron
    # UniformPrior("voc_emergence.omicron.new_voc_seed.start_time", (746.0, 781.0)),  # 5 weeks interval
    # UniformPrior("voc_emergence.omicron.death_protection", (0.8, 1.0)),
    # UniformPrior("voc_emergence.omicron.contact_rate_multiplier", (2, 6)),
    # UniformPrior("voc_emergence.omicron.hosp_protection", (0.8, 1.0)),
    # UniformPrior("voc_emergence.omicron.icu_multiplier", (0.1, 0.75)),
    # UniformPrior("voc_emergence.omicron.relative_active_period", (1.0, 2.5)),
    # UniformPrior("voc_emergence.omicron.relative_latency", (0.01, 0.3)),
    # emergence of delta
    UniformPrior("voc_emergence.wild_type.icu_multiplier", (0.5, 1.5)),
    # UniformPrior("voc_emergence.wild_type.relative_active_period", (1.0, 3.5)),
    # UniformPrior("voc_emergence.wild_type.relative_latency", (0.5, 1.2)),
    # sojourns
    UniformPrior("sojourns.active.proportion_early", (0.5, 1.0)),
    UniformPrior("sojourns.active.total_time", (5, 12)),
    UniformPrior("sojourns.latent.proportion_early", (0.5, 1.0)),
    UniformPrior("sojourns.latent.total_time", (2, 8)),
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
    Region.HO_CHI_MINH_CITY, Models.SM_SIR, build_model, param_set, calibration, plots=plot_spec
)


# from autumn.calibration.proposal_tuning import perform_all_params_proposal_tuning
# perform_all_params_proposal_tuning(project, calibration, priors, n_points=5, relative_likelihood_reduction=0.2)
