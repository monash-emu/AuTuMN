from autumn.core.project import (
    Project,
    ParameterSet,
    load_timeseries,
    build_rel_path,
    use_tuned_proposal_sds,
    get_all_available_scenario_paths,
)
from autumn.calibration import Calibration
from autumn.calibration.priors import UniformPrior
from autumn.calibration.targets import NormalTarget
from autumn.models.sm_sir import base_params, build_model, set_up_random_process
from autumn.settings import Region, Models


def get_scenario_params(pinas_lakas, voc_assumption):
    pinas_lakas_txt = {
        True: "with PinasLakas", False: "with current vacc rates"
    }
    voc_txt = {
        None: "no new VoC",
        "trans": "new more transmissible VoC",
        "ie": "new immune escape VoC"
    }

    description = f"{pinas_lakas_txt[pinas_lakas]} / {voc_txt[voc_assumption]}"
    
    sc_param_dict = {
        "description": description,
        "pinas_lakas": pinas_lakas
    }
    
    if voc_assumption == "trans":
        sc_param_dict["voc_emergence"] = {
            "omicron": {
                "cross_protection": {
                    "new_strain": {
                        "early_reinfection": 1.,
                        "late_reinfection": 0.
                    }
                }
            },
            "new_strain": {
                "new_voc_seed": {
                    "start_time": 1005.,  # 1 Oct 2022
                },
                "contact_rate_multiplier": 2.,
                "immune_escape": 0.
            }
        }
    elif voc_assumption == "ie":
        sc_param_dict["voc_emergence"] = {
            "omicron": {
                "cross_protection": {
                    "new_strain": {
                        "early_reinfection": .5,
                        "late_reinfection": 0.
                    }
                }
            },
            "new_strain": {
                "new_voc_seed": {
                    "start_time": 1005.,  # 1 Oct 2022
                },
                "contact_rate_multiplier": 1.,
                "immune_escape": 0.5
            }
        }

    return sc_param_dict

# Load and configure model parameters.
mle_path = build_rel_path("params/mle-params.yml")
baseline_params = base_params.update(build_rel_path("params/baseline.yml")).update(
    mle_path, calibration_format=True
)

# scenario_dir_path = build_rel_path("params/")
# scenario_paths = get_all_available_scenario_paths(scenario_dir_path)
# scenario_params = [baseline_params.update(p) for p in scenario_paths]

scenario_params = []
for voc_assumption in [None, 'trans', 'ie']:
    for pinas_lakas in [False, True]:    
        if voc_assumption is None and not pinas_lakas:
            continue  # as this is the baseline scenario
        else:
            update_params = get_scenario_params(pinas_lakas, voc_assumption)
            scenario_params.append(baseline_params.update(update_params))

param_set = ParameterSet(baseline=baseline_params, scenarios=scenario_params)

# Load and configure calibration settings.
ts_set = load_timeseries(build_rel_path("timeseries.json"))
priors = [
    UniformPrior("contact_rate", [0.07, 0.15]),
    UniformPrior("sojourns.active.total_time", [2, 6]),
    UniformPrior("infectious_seed", [450, 700]),
    UniformPrior("detect_prop", [0.015, 0.05]),
    # Vaccine-induced immunity
    UniformPrior(
        "immunity_stratification.infection_risk_reduction.low", [0.038, 0.438]
    ),
    UniformPrior(
        "immunity_stratification.infection_risk_reduction.high", [0.438, 0.6]
    ),
    UniformPrior(
        "age_stratification.prop_hospital.source_immunity_protection.low", [0.488, 0.807]
    ),
    UniformPrior(
        "age_stratification.prop_hospital.source_immunity_protection.high", [.85, 0.95]
    ),
    # Hospital-related
    UniformPrior("age_stratification.prop_hospital.multiplier", [0.2, .5]),
    UniformPrior("time_from_onset_to_event.hospitalisation.parameters.mean", [2.0, 5.0]),
    UniformPrior("prop_icu_among_hospitalised", [0.04, 0.10]),
    # UniformPrior("hospital_stay.hospital_all.parameters.mean", [2.0, 8.0]),
    # UniformPrior("hospital_stay.icu.parameters.mean", [3.0, 10.0]),
]

targets = [
    NormalTarget(data=ts_set["ncr_icu_occupancy"].loc[725:]),
    NormalTarget(data=ts_set["ncr_hospital_occupancy"].loc[725:]),
    NormalTarget(data=ts_set["notifications"].loc[725:]),
]

if baseline_params.to_dict()["activate_random_process"]:
    rp_params = baseline_params.to_dict()["random_process"]
    rp = set_up_random_process(rp_params["time"]["start"], rp_params["time"]["end"], rp_params["order"], rp_params["time"]["step"])

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
project = Project(Region.BARMM, Models.SM_SIR, build_model, param_set, calibration, plots=plot_spec)


# from autumn.calibration._tuning import perform_all_params_proposal_tuning
# perform_all_params_proposal_tuning(project, calibration, priors, n_points=20, relative_likelihood_reduction=0.2)
