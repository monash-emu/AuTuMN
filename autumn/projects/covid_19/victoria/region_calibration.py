import numpy as np

from autumn.tools.calibration.priors import UniformPrior, TruncNormalPrior
from autumn.tools.calibration.targets import NormalTarget

# TODO: Add visualisation of raw Google inputs to inputs notebook (minor)
# TODO: Sort out the caps/lower dash/underscore issue in Vic regions naming (minor)
# TODO: Consider a different approach to parameterising vaccine effects (including on hospital and death)
# TODO: Allow for increased severity of Delta (may be needed with vaccination changes)
# TODO: Notebook for visualising outputs after run

# Specify the general features of the calibration
target_start_time = 454

# Median unadjusted posterior contact rate from 2020: 0.0463
priors = [
    UniformPrior(
        "contact_rate",
        (0.1, 0.3), jumping_stdev=0.05
    ),
    # UniformPrior(
    #     "seasonal_force",
    #     (0., 0.3), jumping_stdev=0.05
    # ),
    UniformPrior(
        "vaccination.fully_vaccinated.vacc_reduce_infectiousness",
        (0.1, 0.3)
    ),
    UniformPrior(
        "testing_to_detection.assumed_cdr_parameter",
        (0.1, 0.25), jumping_stdev=0.04
    ),
    TruncNormalPrior(
        "sojourn.compartment_periods_calculated.exposed.total_period",
        mean=6.095798813756773, stdev=0.7810560402997285, trunc_range=(1.0, np.inf), jumping_stdev=0.5
    ),
    TruncNormalPrior(
        "sojourn.compartment_periods_calculated.active.total_period",
        mean=6.431724510638751, stdev=0.6588899585941116, trunc_range=(3.0, np.inf), jumping_stdev=0.4
    ),
]
regional_target_names = (
    "notifications",
    "hospital_admissions"
)
metro_target_names = (
    "notifications",
    # "infection_deaths",
    "hospital_admissions",
    "icu_admissions",
    "hospital_occupancy",
    "icu_occupancy",
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
            NormalTarget(timeseries=ts_set.get(target_name).truncate_start_time(target_start_time))
        )
    return targets
