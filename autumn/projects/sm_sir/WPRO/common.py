from autumn.calibration.priors import UniformPrior
from autumn.calibration.targets import NormalTarget
from autumn.core.project import load_timeseries, build_rel_path


def get_WPRO_priors():
    priors = [
        UniformPrior("contact_rate", (0.03, 0.3)),
        UniformPrior("infectious_seed", (1000, 10000)),
        UniformPrior("age_stratification.cfr.multiplier", (0.01, 0.7)),
        UniformPrior("testing_to_detection.assumed_cdr_parameter", (0.01, 0.1)),
        UniformPrior("sojourns.latent.total_time", (2, 12)),
        UniformPrior("voc_emergence.omicron.new_voc_seed.start_time", (500, 800)),
        UniformPrior("voc_emergence.omicron.contact_rate_multiplier", (0.8, 3))
    ]
    return priors


def get_tartgets(calibration_start_time, country_name, region_name):
    seperator = "/"
    timeseries_file = "timeseries.json"
    time_series_path = build_rel_path(f"{country_name}{seperator}{region_name}{seperator}{timeseries_file}")
    ts_set = load_timeseries(time_series_path)

    infection_deaths_ts = ts_set["infection_deaths"].loc[calibration_start_time:]
    notifications_ts = ts_set["notifications"].loc[calibration_start_time:]
    targets = [
        NormalTarget(infection_deaths_ts),
        NormalTarget(notifications_ts)
    ]
    return targets