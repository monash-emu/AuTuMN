from autumn.calibration.priors import UniformPrior
from autumn.calibration.targets import NormalTarget
from autumn.core.project import load_timeseries, build_rel_path


def get_WPRO_priors():
    priors = [
        UniformPrior("contact_rate", (0.1, 0.3)),
        # UniformPrior("testing_to_detection.assumed_cdr_parameter", (0.01, 0.1)),
        UniformPrior("sojourns.latent.total_time", (5, 20)),
        UniformPrior("voc_emergence.omicron.contact_rate_multiplier", (1.1, 1.3)),
        UniformPrior("voc_emergence.omicron.new_voc_seed.start_time", (520, 675)),
        UniformPrior("voc_emergence.omicron.relative_latency", (0.45, 0.75)),
        UniformPrior("voc_emergence.omicron.relative_active_period", (0.4, 1.3))
    ]
    return priors


def get_tartgets(calibration_start_time, iso3, region):
    seperator = "/"
    timeseries_file = "timeseries.json"
    time_series_path = build_rel_path(f"{iso3}{seperator}{region}{seperator}{timeseries_file}")
    ts_set = load_timeseries(time_series_path)

    infection_deaths_ts = ts_set["infection_deaths"].loc[calibration_start_time:]
    targets = [
        NormalTarget(infection_deaths_ts)
    ]
    return targets
