import random

import pandas as pd

from autumn.db.uncertainty import calculate_mcmc_uncertainty


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
    chains = list(range(3))  # Simulate 3 calibration chains
    runs = list(range(1000))  # 1000 runs per chain
    times = list(range(100))  # 100 timesteps per run

    # Function to calculate uncertainty for.
    def incidence_func(t):
        # Linear func + [0, 1] from uniform distribution
        return t + random.random()

    def foo_func(t):
        # Quadratic func + [0, 1] from uniform distribution
        return t ** 2 + random.random()

    # Build dataframes for database tables.
    do_columns = ["chain", "run", "scenario", "times", "incidence", "foo"]
    do_data = {"chain": [], "run": [], "scenario": [], "times": [], "incidence": [], "foo": []}
    mcmc_columns = ["chain", "run", "loglikelihood", "ap_loglikelihood", "accept", "weight"]
    mcmc_data = {
        "chain": [],
        "run": [],
        "loglikelihood": [],
        "ap_loglikelihood": [],
        "accept": [],
        "weight": [],
    }
    # Create synthetic data
    for chain in chains:
        last_accept_idx = 0
        for run_idx, run in enumerate(runs):
            # Simulate filling the mcmc_run table.
            mcmc_data["chain"].append(chain)
            mcmc_data["run"].append(run)
            mcmc_data["loglikelihood"].append(0)
            mcmc_data["ap_loglikelihood"].append(0)

            is_accepted = random.random() > 0.6
            if not is_accepted:
                accept = 0
                weight = 0
            else:
                accept = 1
                weight = 1
                idx = run_idx - last_accept_idx
                last_accept_idx = run_idx
                if mcmc_data["weight"]:
                    mcmc_data["weight"][-idx] = idx

            mcmc_data["weight"].append(weight)
            mcmc_data["accept"].append(accept)
            for time in times:
                # Simulate filling the derived_outputs table.
                do_data["chain"].append(chain)
                do_data["run"].append(run)
                do_data["scenario"].append(0)
                do_data["times"].append(time)
                do_data["incidence"].append(incidence_func(time))
                do_data["foo"].append(foo_func(time))

    do_df = pd.DataFrame(columns=do_columns, data=do_data)
    mcmc_df = pd.DataFrame(columns=mcmc_columns, data=mcmc_data)
    # Calculate uncertainty from synthetic data.
    unc_df = calculate_mcmc_uncertainty(mcmc_df, do_df, targets)
    # Check that calculated quantiles are correct for incidence.
    for quantile in quantiles:
        mask = (unc_df["quantile"] == quantile) & (unc_df["type"] == "incidence")
        vals = (unc_df[mask]["value"] - unc_df[mask]["time"]).to_numpy()
        assert ((vals > (quantile - 0.07)) * (vals < (quantile + 0.07))).all()

    # Check that calculated quantiles are correct for foo.
    for quantile in quantiles:
        mask = (unc_df["quantile"] == quantile) & (unc_df["type"] == "foo")
        vals = (unc_df[mask]["value"] - unc_df[mask]["time"].apply(lambda t: t ** 2)).to_numpy()
        assert ((vals > (quantile - 0.07)) * (vals < (quantile + 0.07))).all()
