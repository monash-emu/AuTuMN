from autumn.tools.calibration.priors import UniformPrior
from autumn.tools.calibration.targets import NormalTarget

# TODO: Look at contact tracing computed values

# Specify the general features of the calibration
target_start_time = 454
priors = [
    UniformPrior("contact_rate", [0.025, 0.05]),
]


def collate_regional_targets(ts_set):
    """
    Collate all the regional targets using the set of timeseries created in the project.py file from the application
    folder.
    """

    targets = []
    for target_name in ("notifications", "hospital_admissions"):
        targets.append(
            NormalTarget(timeseries=ts_set.get(target_name).truncate_start_time(target_start_time))
        )
    return targets


def collate_metro_targets(ts_set):
    """
    As for function above, except for Metropolitan Melbourne clusters.
    """

    targets = []
    for target_name in ("notifications", "hospital_admissions"):
        targets.append(
            NormalTarget(timeseries=ts_set.get(target_name).truncate_start_time(target_start_time))
        )
    return targets
