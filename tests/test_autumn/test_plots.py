# Test plotting functions so that we know they'll run without crashing.
# Does not check the actual contents of the plots, because it'd be too fiddly and fragile.
import os
import random

from autumn.core.db import Database
from autumn.core.db.uncertainty import calculate_mcmc_uncertainty
from autumn.core.plots.calibration import plot_post_calibration
from autumn.core.plots.uncertainty import plot_uncertainty
from tests.utils import build_synthetic_calibration


def test_plot_post_calibration(tmp_path):
    plot_dir = tmp_path
    mcmc_dir_path = os.path.join(tmp_path, "mcmc")
    os.makedirs(mcmc_dir_path)
    targets = {
        "incidence": {
            "output_key": "incidence",
            "title": "incidence",
            "times": [0],
            "values": [0],
            "quantiles": [0.25, 0.5, 0.75],
        },
        "foo": {
            "output_key": "foo",
            "title": "foo",
            "times": [],
            "values": [],
            "quantiles": [0.25, 0.5, 0.75],
        },
    }

    # A dummy prior to pass postirior checks
    priors = [
        {"param_name": "contact_rate", "distribution": "uniform", "distri_params": [0.01, 0.03]}
    ]

    funcs = [lambda t: 2 * t + random.random(), lambda t: t ** 3 + random.random()]
    # Build data for plotting
    do_df, mcmc_df, params_df = build_synthetic_calibration(
        targets, funcs, chains=2, runs=20, times=20
    )
    chains = set(mcmc_df["chain"].tolist())
    # Create databases for plotting
    for chain in chains:
        db_path = os.path.join(mcmc_dir_path, f"chain-{chain}.db")
        db = Database(db_path)
        db.dump_df("mcmc_run", mcmc_df[mcmc_df["chain"] == chain])
        db.dump_df("mcmc_params", params_df[params_df["chain"] == chain])
        db.dump_df("derived_outputs", do_df[do_df["chain"] == chain])

    # Create plots
    plot_post_calibration(targets, mcmc_dir_path, plot_dir, priors)

    # Check plots - do a super basic check
    expected_files = [
        "loglikelihood-traces.png",
        "acceptance_ratio.png",
        "params-traces",
        "calibration-fit",
        "params-vs-loglikelihood",
        "posteriors",
        # "autocorrelations",
    ]
    for fname in expected_files:
        p = os.path.join(plot_dir, fname)
        assert os.path.exists(p)
        if os.path.isdir(p):
            assert len(os.listdir(p)) > 0


def test_plot_uncertainty(tmp_path):
    """
    Ensure uncertainty plotting code works.
    """
    output_dir = tmp_path
    powerbi_db_path = os.path.join(tmp_path, "powerbi.db")
    targets = {
        "incidence": {
            "output_key": "incidence",
            "title": "incidence",
            "times": [],
            "values": [],
            "quantiles": [0.25, 0.5, 0.75],
        },
        "foo": {
            "output_key": "foo",
            "title": "foo",
            "times": [],
            "values": [],
            "quantiles": [0.25, 0.5, 0.75],
        },
    }
    funcs = [lambda t: 2 * t + random.random(), lambda t: t ** 3 + random.random()]
    # Build data for plotting
    do_df, mcmc_df, _ = build_synthetic_calibration(targets, funcs, chains=2, runs=20, times=20)
    unc_df = calculate_mcmc_uncertainty(mcmc_df, do_df, targets)
    # Create database for plotting
    db = Database(powerbi_db_path)
    db.dump_df("mcmc_run", mcmc_df)
    db.dump_df("derived_outputs", do_df)
    db.dump_df("uncertainty", unc_df)
    # Create plots
    plot_uncertainty(targets, powerbi_db_path, output_dir)
    # Check plots
    expected_foo_path = os.path.join(tmp_path, "foo", "uncertainty-foo-0.png")
    expected_incidence_path = os.path.join(tmp_path, "incidence", "uncertainty-incidence-0.png")
    assert os.path.exists(expected_foo_path)
    assert os.path.exists(expected_incidence_path)
