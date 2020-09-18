import logging

from autumn.constants import Region
from autumn.calibration import Calibration
from autumn.tool_kit.params import load_params, load_targets

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
        "param_name": "contact_rate",
        "distribution": "uniform",
        "distri_params": [1., 10.]
    },
    {
        "param_name": "time_variant_presentation_delay.end_value",
        "distribution": "uniform",
        "distri_params": [.5, 4.]  # roughly CDR in 30-80 %
    },
    {
        "param_name": "user_defined_stratifications.location.adjustments.detection_rate.ebeye",
        "distribution": "uniform",
        "distri_params": [0.5, 2.0],
    },
    {
        "param_name": "user_defined_stratifications.location.adjustments.detection_rate.other",
        "distribution": "uniform",
        "distri_params": [0.5, 2.0],
    },
    {
        "param_name": "extra_params.rr_progression_diabetes",
        "distribution": "uniform",
        "distri_params": [2.25, 5.73],
    },
]
# add latency parameters
for param_name in ['early_activation_rate', 'late_activation_rate']:
    for agegroup in ['age_0', 'age_5', 'age_15']:
        PRIORS.append(
            get_latency_priors_from_epidemics(param_name, agegroup)
        )

targets_to_use = [
    'prevalence_infectiousXlocation_majuro',
    'prevalence_infectiousXlocation_ebeye',
    'percentage_latentXlocation_majuro',
    'notificationsXlocation_majuro',
    'notificationsXlocation_ebeye',
    'notificationsXlocation_other'
]
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
