import pandas as pd

from autumn.tools.project import (
    Project,
    ParameterSet,
    load_timeseries,
    build_rel_path,
    get_all_available_scenario_paths,
    use_tuned_proposal_sds,
)
from autumn.tools.calibration import Calibration
from autumn.tools.calibration.priors import UniformPrior
from autumn.tools.calibration.targets import NormalTarget
from autumn.models.sm_sir import (
    base_params,
    build_model,
    set_up_random_process
)
from autumn.settings import Region, Models

from autumn.projects.covid_19.calibration import COVID_GLOBAL_PRIORS

# Load and configure model parameters
mle_path = build_rel_path("params/mle-params.yml")
scenario_dir_path = build_rel_path("params/")
scenario_paths = get_all_available_scenario_paths(scenario_dir_path)

baseline_params = base_params.update(build_rel_path("params/baseline.yml")).update(
    mle_path, calibration_format=True
)
scenario_params = [baseline_params.update(p) for p in scenario_paths]
param_set = ParameterSet(baseline=baseline_params, scenarios=[])

# Load and configure calibration settings.
ts_set = load_timeseries(build_rel_path("timeseries.json"))

# notifications = ts_set["notifications"].multiple_truncations([[511, 575], [606, 700]])
# truncated from 18th Jul to 28th Jul, then from 28th Aug onwards
notifications = pd.concat(
    [ts_set["notifications"].loc[511:575],
     ts_set["notifications"].loc[606:700]]
)

icu_occupancy = ts_set["icu_occupancy"].loc[640:]  # truncated to 01 Oct 2021
infection_deaths = ts_set["infection_deaths"].loc[556:]  # truncated to 9th Jul 2021

targets = [NormalTarget(notifications),
           NormalTarget(icu_occupancy),
           NormalTarget(infection_deaths)]


priors = [
    # Global COVID priors
    # *COVID_GLOBAL_PRIORS,
    # Starting date
    # UniformPrior("time.start", [455, 485], jumping_stdev=3.0),
    # Regional parameters
    UniformPrior("infectious_seed", [1, 100]),
    UniformPrior("contact_rate", [0.1, 0.25]),
    # Health system-related
    # UniformPrior("clinical_stratification.icu_prop", [0.14, 0.18]),
    # UniformPrior("clinical_stratification.non_sympt_infect_multiplier", [0.15, 1.0]),
    # UniformPrior("clinical_stratification.props.symptomatic.multiplier", [0.6, 1.0]),
    # UniformPrior("hospital_prop_multiplier", [0.25, 0.4]),
    # UniformPrior("infection_fatality.multiplier", [0.7, 1.0]),
    # Detection
    UniformPrior("testing_to_detection.assumed_cdr_parameter", [0.0005, 0.02]),
    UniformPrior("sojourns.latent.total_time", (1, 4.0)),
    # Microdistancing
    # UniformPrior("mobility.microdistancing.behaviour.parameters.max_effect", [0.30, 0.40]),
    # Waning immunity
    # UniformPrior("waning_immunity_duration", (180, 360), jumping_stdev=30.),
    # Vaccination parameters (independent sampling)
    # TruncNormalPrior("vaccination.one_dose.ve_prop_prevent_infection", mean=0.9, stdev=0.02, truc_range=(0.8, 1)),
    # TruncNormalPrior("vaccination.one_dose.ve_sympt_covid", mean=0.5, stdev=0.02, truc_range=(0.4, 0.6)),
    # Partly-waned immunity of vaccine
    # TruncNormalPrior("vaccination.part_waned.ve_sympt_covid", mean=0.5, stdev=0.02, truc_range=(0.4, 0.6)),
    # TruncNormalPrior("vaccination.part_waned.ve_infectiousness", mean=0.5, stdev=0.02, truc_range=(0.2, 0.3)),
    # TruncNormalPrior("vaccination.part_waned.ve_hospitalisation", mean=0.75, stdev=0.02, truc_range=(0.65, 0.85)),
    # TruncNormalPrior("vaccination.part_waned.ve_death", mean=0.8, stdev=0.02, truc_range=(0.7, 0.9))
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


# from autumn.tools.calibration.proposal_tuning import perform_all_params_proposal_tuning
# perform_all_params_proposal_tuning(project, calibration, priors, n_points=50, relative_likelihood_reduction=0.2)
