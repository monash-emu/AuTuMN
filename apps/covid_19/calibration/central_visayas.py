from autumn.constants import Region
from apps.covid_19.calibration import base
from apps.covid_19.calibration.base import \
    provide_default_calibration_params, add_standard_dispersion_parameter, add_case_detection_params_philippines, \
    assign_trailing_weights_to_halves


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


# Notification data:
notification_times = [
44,
68,
72,
82,
86,
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
194,
201,
]

notification_values = [
1,
10,
2,
3,
4,
2,
3,
30,
58,
124,
79,
46,
54,
80,
125,
181,
182,
247,
305,
206,
169,
]

# ICU data:
icu_times = [
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
194,
201,
208,
]

icu_values = [
3,
13,
18,
19,
19,
20,
26,
30,
31,
47,
60,
76,
82,
71,
80,
]

TARGET_OUTPUTS = [
    {
        "output_key": "notifications",
        "years": notification_times,
        "values": notification_values,
        "loglikelihood_distri": "negative_binomial",
        "time_weights": assign_trailing_weights_to_halves(5, notification_times),
    },
        {
        "output_key": "icu_occupancy",
        "years": icu_times,
        "values": icu_values,
        "loglikelihood_distri": "negative_binomial",
        "time_weights": list(range(1, len(icu_times) + 1)),
    }
]


PAR_PRIORS = provide_default_calibration_params()
PAR_PRIORS = add_standard_dispersion_parameter(PAR_PRIORS, TARGET_OUTPUTS, "notifications")
PAR_PRIORS = add_standard_dispersion_parameter(PAR_PRIORS, TARGET_OUTPUTS, "icu_occupancy")
PAR_PRIORS = add_case_detection_params_philippines(PAR_PRIORS)
