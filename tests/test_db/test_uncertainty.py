import random

import pandas as pd

from autumn.db.uncertainty import calculate_mcmc_uncertainty


def test_calculate_mcmc_uncertainty():
    """
    Ensure MCMC uncertainty works for a straight line with uniform noise.
    """
    targets = {
        "incidence": {
            "output_key": "incidence",
            "title": "incidence",
            "times": [],
            "values": [],
            "quantiles": [0.25, 0.5, 0.75],
        }
    }

    times = list(range(100))
    chains = list(range(3))
    runs = list(range(500))

    def func(t):
        # Linear function + [0, 1] from uniform distribution
        return t + random.random()

    do_columns = ["chain", "run", "scenario", "times", "incidence"]
    do_data = {"chain": [], "run": [], "scenario": [], "times": [], "incidence": []}
    mcmc_columns = ["chain", "run", "loglikelihood", "ap_loglikelihood", "accept", "weight"]
    mcmc_data = {
        "chain": [],
        "run": [],
        "loglikelihood": [],
        "ap_loglikelihood": [],
        "accept": [],
        "weight": [],
    }

    for chain in chains:
        last_accept_idx = 0
        for run_idx, run in enumerate(runs):
            # Simulate filling the mcmc_run table.
            is_accepted = random.random() > 0.6
            mcmc_data["chain"].append(chain)
            mcmc_data["run"].append(run)
            mcmc_data["loglikelihood"].append(0)
            mcmc_data["ap_loglikelihood"].append(0)
            mcmc_data["weight"].append(1)
            if not is_accepted:
                accept = 0
            else:
                accept = 1
                idx = run_idx - last_accept_idx
                last_accept_idx = run_idx
                if mcmc_data["accept"]:
                    mcmc_data["accept"][-idx] = idx

            mcmc_data["accept"].append(accept)
            for time in times:
                # Simulate filling the derived_outputs table.
                do_data["chain"].append(chain)
                do_data["run"].append(run)
                do_data["scenario"].append(0)
                do_data["times"].append(time)
                do_data["incidence"].append(func(time))

    do_df = pd.DataFrame(columns=do_columns, data=do_data)
    mcmc_df = pd.DataFrame(columns=mcmc_columns, data=mcmc_data)
    unc_df = calculate_mcmc_uncertainty(mcmc_df, do_df, targets)

    for quantile in [0.25, 0.5, 0.75]:
        mask = unc_df["quantile"] == quantile
        vals = (unc_df[mask]["value"] - unc_df[mask]["time"]).to_numpy()
        assert ((vals > (quantile - 0.05)) * (vals < (quantile + 0.05))).all()
