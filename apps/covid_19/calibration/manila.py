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
        Region.MANILA,
        PAR_PRIORS,
        TARGET_OUTPUTS,
        mode="autumn_mcmc",
    )

# START CALIBRATION VALUES
# Notification data:
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
    194,
    201,
    208,
    215,
]

notification_values = [
    2,
    1,
    2,
    23,
    60,
    112,
    145,
    148,
    197,
    146,
    147,
    148,
    168,
    174,
    224,
    180,
    213,
    244,
    343,
    530,
    917,
    1206,
    1450,
    1638,
    1367,
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
    82,
    161,
    206,
    244,
    226,
    224,
    221,
    226,
    219,
    224,
    236,
    222,
    291,
    359,
    368,
    383,
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
