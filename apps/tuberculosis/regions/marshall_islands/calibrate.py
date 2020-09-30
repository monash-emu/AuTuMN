import logging

from autumn.constants import Region
from autumn.calibration import Calibration
from autumn.tool_kit.params import load_params, load_targets
from autumn.calibration.utils import add_dispersion_param_prior_for_gaussian


from apps.tuberculosis.model import build_model
from apps.tuberculosis.calibration_utils import get_latency_priors_from_epidemics

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
        param_set_name=Region.MARSHALL_ISLANDS,
    )
    calib.run_fitting_algorithm(
        run_mode="autumn_mcmc",
        n_iterations=1e6,
        n_burned=0,
        n_chains=1,
        available_time=max_seconds,
    )


PRIORS = [
    {
        "param_name": "start_population_size",
        "distribution": "uniform",
        "distri_params": [2000, 6000]
    },
    {
        "param_name": "contact_rate",
        "distribution": "uniform",
        "distri_params": [1.5, 2.0]
    },
    {
        "param_name": "late_reactivation_multiplier",
        "distribution": "uniform",
        "distri_params": [.5, 2.]
    },
    {
        "param_name": "time_variant_tb_screening_rate.end_value",
        "distribution": "uniform",
        "distri_params": [.3, .5]
    },
    {
        "param_name": "user_defined_stratifications.location.adjustments.detection_rate.ebeye",
        "distribution": "uniform",
        "distri_params": [0.5, 1.5],
    },
    {
        "param_name": "user_defined_stratifications.location.adjustments.detection_rate.other",
        "distribution": "uniform",
        "distri_params": [0.5, 1.5],
    },
    {
        "param_name": "extra_params.rr_progression_diabetes",
        "distribution": "uniform",
        "distri_params": [2.25, 5.73],
    },
]

targets_to_use = [
    'prevalence_infectiousXlocation_majuro',
    'prevalence_infectiousXlocation_ebeye',
    'percentage_latentXlocation_majuro',
    'notificationsXlocation_majuro',
    'notificationsXlocation_ebeye',
    'population_size',
]

target_sds = {
    'percentage_latentXlocation_majuro': .4,
    'prevalence_infectiousXlocation_majuro': 80.,
    'prevalence_infectiousXlocation_ebeye': 120.,
    'population_size': 2500.,
}

TARGET_OUTPUTS = []
for t_name, t in targets.items():
    if t['output_key'] in targets_to_use:
        TARGET_OUTPUTS.append(
            {
                "output_key": t['output_key'],
                "years": t["times"],
                "values": t["values"],
                "loglikelihood_distri": "normal",
            },
        )
        if t['output_key'] in target_sds:
            TARGET_OUTPUTS[-1]["sd"] = target_sds[t['output_key']]


PRIORS = add_dispersion_param_prior_for_gaussian(PRIORS, TARGET_OUTPUTS)
