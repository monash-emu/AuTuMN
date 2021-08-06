import numpy as np

from autumn.tools.project import TimeSeriesSet
from autumn.tools.calibration.priors import UniformPrior, TruncNormalPrior, BetaPrior
from autumn.tools.calibration.targets import NegativeBinomialTarget, TruncNormalTarget

from autumn.projects.covid_19.calibration import COVID_GLOBAL_PRIORS


def get_philippies_calibration_settings(ts_set: TimeSeriesSet):
    """
    Returns standard priors and targets for a Phillipines COVID model calibration.
    """
    # Set a time before which targets are ignored
    cutoff_time = 366  # 31 Dec 2020

    # Notifications with increasing weights for the last 90 days
    notifications_ts = ts_set.get("notifications").truncate_start_time(cutoff_time)
    n = len(notifications_ts.times)
    max_weight = 10.
    notification_weights = [1.0 for _ in range(n - 90)] + [1.0 + (i + 1) * (max_weight - 1.) / 90 for i in range(90)]

    # Only use most recent datapoint for icu occupancy
    icu_occupancy_ts = ts_set.get("icu_occupancy")[-1]

    # Use 1 Jan 2021 and most recent datapoint for accumulated deaths
    accum_deaths_ts = ts_set.get("infection_deaths", name="accum_deaths").truncate_start_time(cutoff_time)
    accum_deaths_ts.times = [accum_deaths_ts.times[0], accum_deaths_ts.times[-1]]
    accum_deaths_ts.values = [accum_deaths_ts.values[0], accum_deaths_ts.values[-1]]

    targets = [
        TruncNormalTarget(notifications_ts, trunc_range=[0, np.inf], time_weights=notification_weights),
        TruncNormalTarget(icu_occupancy_ts, trunc_range=[0, np.inf]),
        TruncNormalTarget(accum_deaths_ts, trunc_range=[0, np.inf]),
    ]

    priors = [
        # Philippines country-wide priors
        UniformPrior("contact_rate", [0.03, 0.05]),
        UniformPrior("testing_to_detection.assumed_cdr_parameter", [0.02, 0.03]),
        UniformPrior("infection_fatality.multiplier", [0.5, 3.]),
        UniformPrior("clinical_stratification.props.hospital.multiplier", [0.5, 3.]),
        UniformPrior("voc_emergence.delta.contact_rate_multiplier", [2., 3.]),

        # Vaccination parameters (independent sampling)
        UniformPrior("vaccination.vacc_prop_prevent_infection", [0, 1], sampling="lhs"),
        BetaPrior("vaccination.overall_efficacy", mean=0.7, ci=[0.5, 0.9], sampling="lhs"),
    ]
    return targets, priors
