import random


from autumn.core.db.uncertainty import calculate_mcmc_uncertainty
from tests.utils import build_synthetic_calibration


def test_calculate_mcmc_uncertainty():
    """
    Ensure MCMC uncertainty works for a straight line with uniform noise.
    """
    # Requested uncertainty quantiles.
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    # Calibration targets
    targets = {
        "incidence": {
            "output_key": "incidence",
            "title": "incidence",
            "times": [],
            "values": [],
            "quantiles": quantiles,
        },
        "foo": {
            "output_key": "foo",
            "title": "foo",
            "times": [],
            "values": [],
            "quantiles": quantiles,
        },
    }
    funcs = [_linear_func, _quadratic_func]
    do_df, mcmc_df, _ = build_synthetic_calibration(targets, funcs, chains=3, runs=1000, times=100)

    # Calculate uncertainty from synthetic data.
    unc_df = calculate_mcmc_uncertainty(mcmc_df, do_df, targets)
    # Check that calculated quantiles are correct for incidence.
    TOLERANCE = 0.1

    for quantile in quantiles:
        mask = (unc_df["quantile"] == quantile) & (unc_df["type"] == "incidence")
        vals = (unc_df[mask]["value"] - unc_df[mask]["time"]).to_numpy()
        assert ((vals > (quantile - TOLERANCE)) * (vals < (quantile + TOLERANCE))).all()

    # Check that calculated quantiles are correct for foo.
    for quantile in quantiles:
        mask = (unc_df["quantile"] == quantile) & (unc_df["type"] == "foo")
        vals = (unc_df[mask]["value"] - unc_df[mask]["time"].apply(lambda t: t ** 2)).to_numpy()
        assert ((vals > (quantile - TOLERANCE)) * (vals < (quantile + TOLERANCE))).all()


# Functions to calculate uncertainty for.
def _linear_func(t):
    # Linear func + [0, 1] from uniform distribution
    return t + random.random()


def _quadratic_func(t):
    # Quadratic func + [0, 1] from uniform distribution
    return t ** 2 + random.random()
