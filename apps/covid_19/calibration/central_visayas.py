from autumn.constants import Region
from apps.covid_19.calibration import base
from apps.covid_19.calibration.base import (
    provide_default_calibration_params,
    add_standard_dispersion_parameter,
    add_standard_philippines_params,
    assign_trailing_weights_to_halves,
)


def run_calibration_chain(max_seconds: int, run_id: int, num_chains: int):
    base.run_calibration_chain(
        max_seconds,
        run_id,
        num_chains,
        Region.CENTRAL_VISAYAS,
        PAR_PRIORS,
        TARGET_OUTPUTS,
        mode="autumn_mcmc",
    )

# START CALIBRATION VALUES
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
    208,
    215,
]

notification_values = [
    1,
    11,
    2,
    3,
    4,
    2,
    3,
    30,
    59,
    124,
    79,
    46,
    55,
    82,
    127,
    186,
    185,
    248,
    338,
    267,
    230,
    228,
    149,
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
    215,
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
    79,
    72,
]

# END CALIBRATION VALUES

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
    },
]


PAR_PRIORS = provide_default_calibration_params()
PAR_PRIORS = add_standard_dispersion_parameter(PAR_PRIORS, TARGET_OUTPUTS, "notifications")
PAR_PRIORS = add_standard_dispersion_parameter(PAR_PRIORS, TARGET_OUTPUTS, "icu_occupancy")
PAR_PRIORS = add_standard_philippines_params(PAR_PRIORS)
