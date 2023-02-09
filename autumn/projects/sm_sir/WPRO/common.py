from autumn.calibration.priors import UniformPrior
from autumn.core.project import load_timeseries, build_rel_path
from autumn.models.sm_covid.stratifications.strains import get_first_variant_report_date
from autumn.settings.constants import COVID_BASE_DATETIME


def get_WPRO_priors(variant_times):
    priors = [
        UniformPrior("infectious_seed", (1000, 10000)),
        UniformPrior("testing_to_detection.assumed_cdr_parameter", (0.01, 0.1)),
        UniformPrior("sojourns.latent.total_time", (4, 15)),
        UniformPrior(
            "voc_emergence.omicron.new_voc_seed.start_time",
            (variant_times[1] - 150, variant_times[1] + 100),
        ),
        UniformPrior("voc_emergence.omicron.contact_rate_multiplier", (0.8, 3.5)),
    ]
    return priors


def get_AUS_priors():
    priors = [
        UniformPrior("infectious_seed", (1000, 10000)),
        UniformPrior("testing_to_detection.assumed_cdr_parameter", (0.01, 0.1)),
        UniformPrior("sojourns.latent.total_time", (4, 15)),
        UniformPrior("voc_emergence.ba_2.new_voc_seed.start_time", (690, 750)),
        UniformPrior("voc_emergence.ba_2.contact_rate_multiplier", (0.8, 3.5)),
    ]
    return priors


def get_JPN_priors():
    priors = [
        UniformPrior("infectious_seed", (1000, 10000)),
        UniformPrior("testing_to_detection.assumed_cdr_parameter", (0.01, 0.1)),
        UniformPrior("sojourns.latent.total_time", (4, 15)),
        UniformPrior("voc_emergence.ba_2.new_voc_seed.start_time", (890, 950)),
        UniformPrior("voc_emergence.ba_2.contact_rate_multiplier", (1.0, 2.0)),
    ]
    return priors

def get_SIN_priors():
    priors = [
        UniformPrior("infectious_seed", (1000, 10000)),
        UniformPrior("testing_to_detection.assumed_cdr_parameter", (0.01, 0.1)),
        UniformPrior("sojourns.latent.total_time", (4, 15)),
        UniformPrior("voc_emergence.ba_1.contact_rate_multiplier", (0.8, 1.5)),
        UniformPrior("voc_emergence.ba_1.new_voc_seed.start_time", (580, 650)),
        UniformPrior("voc_emergence.ba_2.contact_rate_multiplier", (1.0, 2.0)),
        UniformPrior("voc_emergence.ba_2.new_voc_seed.start_time", (655, 700)),
    ]
    return priors

def get_MNG_priors():
    priors = [
        UniformPrior("infectious_seed", (1000, 10000)),
        UniformPrior("testing_to_detection.assumed_cdr_parameter", (0.01, 0.1)),
        UniformPrior("sojourns.latent.total_time", (4, 15)),
        UniformPrior("voc_emergence.ba_1.contact_rate_multiplier", (0.8, 1.5)),
        UniformPrior("voc_emergence.ba_1.new_voc_seed.start_time", (450, 550)),
        UniformPrior("voc_emergence.ba_2.contact_rate_multiplier", (1.0, 2.0)),
        UniformPrior("voc_emergence.ba_2.new_voc_seed.start_time", (600, 700)),
    ]
    return priors


def get_targets(calibration_start_time, country_name, region_name):
    seperator = "/"
    timeseries_file = "timeseries.json"
    time_series_path = build_rel_path(
        f"{country_name}{seperator}{region_name}{seperator}{timeseries_file}"
    )
    ts_set = load_timeseries(time_series_path)

    return ts_set


def variant_start_time(variants: list, region_name: str):
    variant_times = []
    for variant in variants:
        variant_first_time = get_first_variant_report_date(variant, region_name.title())
        first_report_date_as_int = (variant_first_time - COVID_BASE_DATETIME).days
        variant_times.append(first_report_date_as_int)
    return variant_times
