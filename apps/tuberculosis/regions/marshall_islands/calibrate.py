import logging

from autumn.constants import Region
from autumn.calibration import Calibration
from autumn.tool_kit.params import load_params, load_targets
from autumn.calibration.utils import add_dispersion_param_prior_for_gaussian


from apps.tuberculosis.model import build_model
from apps.tuberculosis.calibration_utils import (
    get_latency_priors_from_epidemics,
    get_natural_history_priors_from_cid,
)

targets = load_targets("tuberculosis", Region.MARSHALL_ISLANDS)


def run_calibration_chain(max_seconds: int, run_id: int, num_chains: int):
    params = load_params("tuberculosis", Region.MARSHALL_ISLANDS)
    calib = Calibration(
        "tuberculosis",
        build_model,
        params,
        PRIORS,
        TARGET_OUTPUTS,
        run_id,
        num_chains,
        region_name=Region.MARSHALL_ISLANDS,
        initialisation_type=params["default"]["metropolis_initialisation"]
    )
    calib.run_fitting_algorithm(
        run_mode="autumn_mcmc",
        n_iterations=1e6,
        n_burned=0,
        n_chains=1,
        available_time=max_seconds,
        haario_scaling_factor=params["default"]["haario_scaling_factor"],
    )


PRIORS = [
    {
        "param_name": "start_population_size",
        "distribution": "uniform",
        "distri_params": [200, 600],
    },
    {"param_name": "contact_rate", "distribution": "uniform", "distri_params": [0.2, 1.]},
    {
        "param_name": "late_reactivation_multiplier",
        "distribution": "uniform",
        "distri_params": [0.3, 3.0],
    },
    {
        "param_name": "time_variant_tb_screening_rate.max_change_time",
        "distribution": "uniform",
        "distri_params": [2000.0, 2020.0],
    },
    {
        "param_name": "time_variant_tb_screening_rate.maximum_gradient",
        "distribution": "uniform",
        "distri_params": [0.07, 0.1],
    },
    {
        "param_name": "time_variant_tb_screening_rate.end_value",
        "distribution": "uniform",
        "distri_params": [0.4, 0.55],
    },
    {
        "param_name": "user_defined_stratifications.location.adjustments.detection_rate.ebeye",
        "distribution": "uniform",
        "distri_params": [1.3, 2.0],  # [.5, 1.5]
    },
    {
        "param_name": "user_defined_stratifications.location.adjustments.detection_rate.other",
        "distribution": "uniform",
        "distri_params": [0.5, 1.5],
    },
    {
        "param_name": "extra_params.rr_progression_diabetes",
        "distribution": "uniform",
        "distri_params": [2., 5.],
    },
    {
        "param_name": "rr_infection_recovered",
        "distribution": "uniform",
        "distri_params": [0.5, 2.0],
    },
    {
        "param_name": "pt_efficacy",
        "distribution": "uniform",
        "distri_params": [0.8, 0.85],
    },
    {
        "param_name": "awareness_raising.relative_screening_rate",
        "distribution": "uniform",
        "distri_params": [1., 1.5],
    }
]

# Add uncertainty around natural history using our CID estimates
for param_name in ["infect_death_rate", "self_recovery_rate"]:
    for organ in ["smear_positive", "smear_negative"]:
        PRIORS.append(get_natural_history_priors_from_cid(param_name, organ))


targets_to_use = [
    "prevalence_infectiousXlocation_majuro",
    "prevalence_infectiousXlocation_ebeye",
    "percentage_latentXlocation_majuro",
    "notificationsXlocation_majuro",
    "notificationsXlocation_ebeye",
    "population_size",
]

target_sds = {
    "percentage_latentXlocation_majuro": 10,
    "prevalence_infectiousXlocation_majuro": 80.0,
    "prevalence_infectiousXlocation_ebeye": 120.0,
    "population_size": 2500.0,
}

TARGET_OUTPUTS = []
for t_name, t in targets.items():
    if t["output_key"] in targets_to_use:
        TARGET_OUTPUTS.append(
            {
                "output_key": t["output_key"],
                "years": t["times"],
                "values": t["values"],
                "loglikelihood_distri": "normal",
            },
        )
        if t["output_key"] in target_sds:
            TARGET_OUTPUTS[-1]["sd"] = target_sds[t["output_key"]]


PRIORS = add_dispersion_param_prior_for_gaussian(PRIORS, TARGET_OUTPUTS)
