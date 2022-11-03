from autumn.calibration.priors import UniformPrior
from autumn.calibration.targets import NormalTarget
from autumn.core.project import load_timeseries, build_rel_path
from autumn.models.sm_covid.stratifications.strains import get_first_variant_report_date
from autumn.settings.constants import COVID_BASE_DATETIME

def get_WPRO_priors(variant_times):
    priors = [
        UniformPrior("infectious_seed", (1000, 10000)),
        UniformPrior("testing_to_detection.assumed_cdr_parameter", (0.01, 0.1)),
        UniformPrior("sojourns.latent.total_time", (2, 12)),
        UniformPrior("voc_emergence.omicron.new_voc_seed.start_time", (variant_times[1]-150, variant_times[1]+100)),
        UniformPrior("voc_emergence.omicron.contact_rate_multiplier", (0.8, 3.5))
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


def variant_start_time(variants: list, region_name: str):
    variant_times = []
    for variant in variants:
        variant_first_time = get_first_variant_report_date(variant, region_name.title())
        first_report_date_as_int = (variant_first_time - COVID_BASE_DATETIME).days
        variant_times.append(first_report_date_as_int)
    return variant_times
