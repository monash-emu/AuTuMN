from autumn.constants import Region
from apps.covid_19.calibration import base
from apps.covid_19.calibration.base import provide_default_calibration_params


def run_calibration_chain(max_seconds: int, run_id: int, num_chains: int):
    base.run_calibration_chain(
        max_seconds,
        run_id,
        num_chains,
        Region.CENTRAL_VISAYAS,
        PAR_PRIORS,
        TARGET_OUTPUTS,
        mode="autumn_mcmc",
        _multipliers=MULTIPLIERS,
    )


MULTIPLIERS = {
    "prevXlateXclinical_icuXamong": 7957050.0
}  # to get absolute pop size instead of proportion

PAR_PRIORS = provide_default_calibration_params()

PAR_PRIORS += [
    {
        "param_name": "time_variant_detection.maximum_gradient",  # shape parameter
        "distribution": "uniform",
        "distri_params": [0.05, 0.1],
    },
    {
        "param_name": "time_variant_detection.max_change_time",  # inflection point
        "distribution": "uniform",
        "distri_params": [70.0, 110.0],
    },
    {
        "param_name": "time_variant_detection.end_value",  # upper asymptote
        "distribution": "uniform",
        "distri_params": [0.10, 0.90],
    },
    # Add extra params for negative binomial likelihood
    {
        "param_name": "notifications_dispersion_param",
        "distribution": "uniform",
        "distri_params": [0.1, 5.0],
    },
    # parameters to derive age-specific IFRs
    {
        "param_name": "ifr_double_exp_model_params.k",
        "distribution": "uniform",
        "distri_params": [6., 14.],
    },
    {
        "param_name": "ifr_double_exp_model_params.last_representative_age",
        "distribution": "uniform",
        "distri_params": [75., 85.],
    },
]

# notification data:
notification_times = [
68,
72,
82,
87,
96,
103,
110,
117,
124,
131,
138,
145,
152,
159,
166,
173,
180,
]

notification_values = [
2,
1,
21,
10,
7,
139,
343,
586,
661,
60,
251,
749,
924,
677,
1867,
1644,
2128,
]

TARGET_OUTPUTS = [
    {
        "output_key": "notifications",
        "years": notification_times,
        "values": notification_values,
        "loglikelihood_distri": "negative_binomial",
        "time_weights": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1., 1., 1., 1., 1., 1.]
    },
]
