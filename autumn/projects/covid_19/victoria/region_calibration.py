from autumn.tools.calibration.priors import UniformPrior
from autumn.tools.calibration.targets import NormalTarget

# TODO: Look at contact tracing computed values

target_start_time = 454
priors = [
    UniformPrior("contact_rate", [0.025, 0.05]),
]


def collate_regional_targets(ts_set):
    notifications_ts = ts_set.get("notifications").truncate_start_time(target_start_time)
    hospital_admissions_ts = ts_set.get("hospital_admissions").truncate_start_time(target_start_time)
    return [
        NormalTarget(timeseries=notifications_ts),
        NormalTarget(timeseries=hospital_admissions_ts)
    ]


def collate_metro_targets(ts_set):
    notifications_ts = ts_set.get("notifications").truncate_start_time(target_start_time)
    hospital_admissions_ts = ts_set.get("hospital_admissions").truncate_start_time(target_start_time)
    return [
        NormalTarget(timeseries=notifications_ts),
        NormalTarget(timeseries=hospital_admissions_ts)
    ]
