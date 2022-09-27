from autumn.calibration import Calibration
from autumn.calibration.priors import UniformPrior
from autumn.calibration.targets import NormalTarget
from autumn.core.project import (
    ParameterSet,
    Project,
    build_rel_path,
    get_all_available_scenario_paths,
    load_timeseries,
    use_tuned_proposal_sds,
)
from autumn.models.sm_sir import base_params, build_model, set_up_random_process
from autumn.settings import Models, Region

"""
THIS FUNCTION HAS BEEN TOTALLY RUINED BY THE VACCINATION RESTRUCTURE
"""


def get_scenario_params(n_boosters, target, voc_emerge):
    doses_txt = {200000: "200K", 500000: "500K", 1000000: "1M"}
    target_txt = {True: "targeted", False: "uniform"}
    voc_txt = {True: "with immune escape VoC", False: "no new VoC"}

    description = f"{doses_txt[n_boosters]} boosters per mth / {target_txt[target]} allocation / {voc_txt[voc_emerge]}"

    sc_param_dict = {
        "description": description,
        # "future_monthly_booster_rate": n_boosters,
    }

    # if target:
    # sc_param_dict["future_booster_age_allocation"] = [60, 50, 25, 15, 0]

    if voc_emerge:
        sc_param_dict["voc_emergence"] = {
            "omicron": {
                "cross_protection": {
                    "new_strain": {"early_reinfection": 0.0, "late_reinfection": 0.0}
                }
            },
            "new_strain": {
                "new_voc_seed": {
                    "start_time": 944.0,  # 1 Aug 2022
                },
                "contact_rate_multiplier": 1.0,
                "immune_escape": 1.0,
            },
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

# scenario_params = []
# for n_boosters in [200000, 500000, 1000000]:
#     for target in [False, True]:
#         for voc_emerge in [False, True]:

#             if n_boosters == 200000 and not target and not voc_emerge:
#                 continue  # as this is the baseline scenario
#             else:
#                 update_params = get_scenario_params(n_boosters, target, voc_emerge)
#                 scenario_params.append(baseline_params.update(update_params))

param_set = ParameterSet(baseline=baseline_params)

# Load and configure calibration settings.
ts_set = load_timeseries(build_rel_path("timeseries.json"))
priors = [
    UniformPrior("contact_rate", [0.07, 0.20]),
    UniformPrior("sojourns.active.total_time", [4, 10]),
    UniformPrior("infectious_seed", [1, 400]),
    UniformPrior("testing_to_detection.assumed_cdr_parameter", [0.005, 0.015]),
    # Vaccine-induced immunity
    UniformPrior("immunity_stratification.infection_risk_reduction.low", [0.038, 0.438]),
    UniformPrior("immunity_stratification.infection_risk_reduction.high", [0.438, 0.6]),
    UniformPrior("age_stratification.prop_hospital.source_immunity_protection.low", [0.488, 0.807]),
    UniformPrior("age_stratification.prop_hospital.source_immunity_protection.high", [0.85, 0.95]),
    # Hospital-related
    UniformPrior("age_stratification.prop_hospital.multiplier", [0.5, 1.5]),
    UniformPrior("time_from_onset_to_event.hospitalisation.parameters.mean", [2.0, 7.0]),
    UniformPrior("prop_icu_among_hospitalised", [0.05, 0.20]),
    UniformPrior("hospital_stay.hospital_all.parameters.mean", [2.0, 8.0]),
    UniformPrior("hospital_stay.icu.parameters.mean", [3.0, 10.0]),
]

targets = [
    NormalTarget(data=ts_set["icu_occupancy"].loc[725:]),
    NormalTarget(data=ts_set["hospital_occupancy"].loc[725:]),
    NormalTarget(data=ts_set["notifications"].loc[725:]),
]

if baseline_params.to_dict()["activate_random_process"]:
    rp_params = baseline_params.to_dict()["random_process"]
    rp = set_up_random_process(
        rp_params["time"]["start"],
        rp_params["time"]["end"],
        rp_params["order"],
        rp_params["time"]["step"],
    )

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
    Region.WESTERN_VISAYAS, Models.SM_SIR, build_model, param_set, calibration, plots=plot_spec
)


# from autumn.calibration._tuning import perform_all_params_proposal_tuning
# perform_all_params_proposal_tuning(project, calibration, priors, n_points=20, relative_likelihood_reduction=0.2)
