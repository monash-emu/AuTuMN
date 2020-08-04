from autumn.constants import Region
from apps.covid_19.calibration import base
from apps.covid_19.calibration.base import \
    provide_default_calibration_params, add_standard_dispersion_parameter, add_standard_philippines_params, \
    assign_trailing_weights_to_halves


def run_calibration_chain(max_seconds: int, run_id: int, num_chains: int):
    base.run_calibration_chain(
        max_seconds,
        run_id,
        num_chains,
        Region.PHILIPPINES,
        PAR_PRIORS,
        TARGET_OUTPUTS,
        mode="autumn_mcmc",
    )


# Notification data:
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
187,
194,
201,
]

notification_values = [
3,
1,
2,
25,
72,
155,
205,
206,
271,
225,
245,
324,
333,
312,
410,
395,
504,
577,
753,
1115,
1720,
1775,
1902,
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
136,
329,
439,
512,
479,
446,
453,
455,
409,
439,
475,
447,
543,
634,
685,
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
PAR_PRIORS = add_standard_philippines_params(PAR_PRIORS)
