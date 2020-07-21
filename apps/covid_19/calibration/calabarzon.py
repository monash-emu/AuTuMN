from autumn.constants import Region
from apps.covid_19.calibration import base


def run_calibration_chain(max_seconds: int, run_id: int, num_chains: int):
    base.run_calibration_chain(
        max_seconds,
        run_id,
        num_chains,
        Region.CALABARZON,
        PAR_PRIORS,
        TARGET_OUTPUTS,
        mode="autumn_mcmc",
        _multipliers=MULTIPLIERS,
    )


MULTIPLIERS = {
    "prevXlateXclinical_icuXamong": 16057300.0
}  # to get absolute pop size instead of proportion


PAR_PRIORS = [
    {
        "param_name": "contact_rate", 
        "distribution": "uniform", 
        "distri_params": [0.010, 0.045],
        },
    {
        "param_name": "start_time", 
        "distribution": "uniform", 
        "distri_params": [0.0, 40.0],
        },
    # Add extra params for negative binomial likelihood
    {
        "param_name": "notifications_dispersion_param",
        "distribution": "uniform",
        "distri_params": [0.1, 5.0],
    },
    {
        "param_name": "compartment_periods_calculated.incubation.total_period",
        "distribution": "gamma",
        "distri_mean": 5.0,
        "distri_ci": [4.4, 5.6],
    },
    {
        "param_name": "compartment_periods_calculated.total_active.total_period",
        "distribution": "gamma",
        "distri_mean": 7.0,
        "distri_ci": [4.5, 9.5],
    },
]

# notification data:
notification_times = [
44,
53,
61,
68,
75,
82,
89,
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
2,
1,
12,
44,
148,
260,
262,
257,
148,
109,
97,
103,
123,
237,
171,
223,
286,
389,
655,
]

TARGET_OUTPUTS = [
    {
        "output_key": "notifications",
        "years": notification_times,
        "values": notification_values,
        "loglikelihood_distri": "negative_binomial",
    },
]
