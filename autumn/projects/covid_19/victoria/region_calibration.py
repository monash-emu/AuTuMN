import numpy as np

from autumn.tools.calibration.priors import UniformPrior, TruncNormalPrior, BetaPrior
from autumn.tools.calibration.targets import NormalTarget

# TODO: Allow for increased severity of Delta (may be needed with vaccination changes)
# TODO: See if we can get deaths as a target too

# Specify the general features of the calibration
target_start_time = 454

# Median unadjusted posterior contact rate from 2020: 0.0463
priors = [
    UniformPrior(
        "contact_rate",
        (0.05, 0.15), jumping_stdev=0.05
    ),
    UniformPrior(
        "vic_2021_seeding.seed_time",
        (560., 580.), jumping_stdev=1.,
    ),
    UniformPrior(
        "testing_to_detection.assumed_cdr_parameter",
        (0.05, 0.18), jumping_stdev=0.04
    ),
    TruncNormalPrior(
        "sojourn.compartment_periods_calculated.exposed.total_period",
        mean=6.095798813756773, stdev=0.7810560402997285, trunc_range=(1.0, np.inf), jumping_stdev=0.5
    ),
    TruncNormalPrior(
        "sojourn.compartment_periods_calculated.active.total_period",
        mean=6.431724510638751, stdev=0.6588899585941116, trunc_range=(3.0, np.inf), jumping_stdev=0.4
    ),
    BetaPrior(
        "vaccination.fully_vaccinated.ve_infectiousness",
        mean=0.32, ci=(0.2, 0.44),
    ),
    BetaPrior(
        "vaccination.fully_vaccinated.ve_prop_prevent_infection",
        mean=0.95, ci=(0.9, 0.98),
    ),
]

regional_target_names = (
    "notifications",
)
metro_target_names = (
    "notifications",
    "hospital_admissions",
    "icu_admissions",
)


def collate_regional_targets(ts_set):
    """
    Collate all the regional targets using the set of timeseries created in the project.py file from the application
    folder.
    """

    targets = []
    for target_name in regional_target_names:
        targets.append(
            NormalTarget(timeseries=ts_set.get(target_name).truncate_start_time(target_start_time))
        )
    return targets


def collate_metro_targets(ts_set):
    """
    As for function above, except for Metropolitan Melbourne clusters.
    """

    targets = []
    for target_name in metro_target_names:
        targets.append(
            NormalTarget(timeseries=ts_set.get(target_name).truncate_times(target_start_time, 660))
        )
    return targets
