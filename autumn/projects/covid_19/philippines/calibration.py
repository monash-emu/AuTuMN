import numpy as np

from autumn.tools.project import TimeSeriesSet
from autumn.tools.calibration.priors import UniformPrior, TruncNormalPrior, BetaPrior
from autumn.tools.calibration.targets import NegativeBinomialTarget

from autumn.projects.covid_19.calibration import COVID_GLOBAL_PRIORS


def get_philippies_calibration_settings(ts_set: TimeSeriesSet):
    """
    Returns standard priors and targets for a Phillipines COVID model calibration.
    """
    # Notifications with weights for the last 60 days
    notifications_ts = ts_set.get("notifications").truncate_start_time(100)
    n = len(notifications_ts.times)
    notification_weights = [1.0 for _ in range(n - 60)] + [5.0 for _ in range(60)]

    # Only use most recent datapoint for icu occupancy and accumulated deaths
    icu_occupancy_ts = ts_set.get("icu_occupancy")[-1]
    accum_deaths_ts = ts_set.get("infection_deaths", name="accum_deaths")[-1]
    targets = [
        NegativeBinomialTarget(notifications_ts, time_weights=notification_weights),
        NegativeBinomialTarget(icu_occupancy_ts),
        NegativeBinomialTarget(accum_deaths_ts),
    ]
    priors = [
        # Global COVID priors
        *COVID_GLOBAL_PRIORS,
        # Dispersion parameters for targets
        *[UniformPrior(f"{t.timeseries.name}_dispersion_param", [0.1, 5.0]) for t in targets],
        # Philippines country-wide priors
        UniformPrior("contact_rate", [0.02, 0.04]),
        UniformPrior("testing_to_detection.assumed_cdr_parameter", [0.02, 0.20]),
        UniformPrior("mobility.microdistancing.behaviour.parameters.max_effect", [0.1, 0.6]),
        UniformPrior("infectious_seed", [1.0, 300.0]),
        TruncNormalPrior(
            "clinical_stratification.props.symptomatic.multiplier",
            mean=1.0,
            stdev=0.2,
            trunc_range=[0.5, np.inf],
        ),
        TruncNormalPrior(
            "clinical_stratification.props.hospital.multiplier",
            mean=1.0,
            stdev=0.5,
            trunc_range=[0.5, np.inf],
        ),
        TruncNormalPrior(
            "infection_fatality.multiplier", mean=1.0, stdev=0.4, trunc_range=[0.5, np.inf]
        ),
        # Between 1 Dec 2020 and 30 June 2021
        UniformPrior("voc_emergence.voc_strain(0).voc_components.start_time", [280, 547]),
        # Using reported 95 CI from Pearson et al.
        UniformPrior("voc_emergence.voc_strain(0).voc_components.contact_rate_multiplier", [1.2, 2.1]),
        UniformPrior(
            "mobility.microdistancing.behaviour_adjuster.parameters.lower_asymptote", [0.8, 1.0]
        ),
        # 1 Jan - 28 Feb
        UniformPrior(
            "mobility.microdistancing.behaviour_adjuster.parameters.inflection_time", [367, 425]
        ),
        # Vaccination parameters (independent sampling)
        UniformPrior("vaccination.vacc_prop_prevent_infection", [0, 1], sampling="lhs"),
        BetaPrior("vaccination.overall_efficacy", mean=0.7, ci=[0.5, 0.9], sampling="lhs"),
    ]
    return targets, priors
