from autumn.constants import Region
from apps.covid_19.calibration import base


def run_calibration_chain(max_seconds: int, run_id: int, num_chains: int):
    base.run_calibration_chain(
        max_seconds,
        run_id,
        num_chains,
        Region.MANILA,
        PAR_PRIORS,
        TARGET_OUTPUTS,
        mode="autumn_mcmc",
    )


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
        "param_name": "compartment_periods_calculated.total_infectious.total_period",
        "distribution": "gamma",
        "distri_mean": 7.0,
        "distri_ci": [4.5, 9.5],
    },
]

# notification data:
notification_times = [
44,
48,
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
187,
]

notification_values = [
2,
1,
4,
51,
365,
754,
1143,
952,
1176,
882,
890,
889,
930,
922,
1624,
1040,
1295,
1410,
2219,
3405,
3151,
]

TARGET_OUTPUTS = [
    {
        "output_key": "notifications",
        "years": notification_times,
        "values": notification_values,
        "loglikelihood_distri": "negative_binomial",
    },
]
