from autumn.calibration.priors import UniformPrior
from autumn.calibration.targets import NormalTarget
from autumn.core.project import load_timeseries, build_rel_path


def get_WPRO_priors():
    priors = [
        UniformPrior("contact_rate", (0.03, 0.3)),
        UniformPrior("infectious_seed", (50, 500)),
        UniformPrior("age_stratification.cfr.multiplier", (0.01, 0.7))
    ]
    return priors


def get_tartgets(calibration_start_time, iso3, region):
    seperator = "/"
    timeseries_file = "timeseries.json"
    time_series_path = build_rel_path(f"{iso3}{seperator}{region}{seperator}{timeseries_file}")
    ts_set = load_timeseries(time_series_path)

    infection_deaths_ts = ts_set["infection_deaths"].loc[calibration_start_time:]
    notifications_ts = ts_set["notifications"].loc[calibration_start_time:]
    targets = [
        NormalTarget(infection_deaths_ts),
    ]
    return targets
