import os
import logging
from typing import List


import numpy as np
import pandas as pd


from autumn.db.database import Database


logger = logging.getLogger(__name__)


def add_uncertainty_quantiles(database_path: str, targets: dict):
    """
    Add an uncertainty table to a given database, based on mcmc_run and derived_outputs.
    The table will have columns scenario/type/time/quantile/value.
    """
    logger.info("Calculating uncertainty for %s", database_path)
    db = Database(database_path)
    if "uncertainty" in db.table_names():
        logger.info(
            "Deleting existing uncertainty table in %s", database_path,
        )
        db.engine.execute(f"DELETE FROM uncertainty")

    logger.info("Loading data into memory")
    mcmc_df = db.query("mcmc_run")
    do_df = db.query("derived_outputs")
    logger.info("Calculating uncertainty")
    uncertainty_df = calculate_mcmc_uncertainty(mcmc_df, do_df, targets)
    db.dump_df("uncertainty", uncertainty_df)
    logger.info("Finished writing uncertainties")


def calculate_mcmc_uncertainty(
    mcmc_df: pd.DataFrame, do_df: pd.DataFrame, targets: dict
) -> pd.DataFrame:
    """
    Calculate quantiles from a table of weighted values.
    See calc_mcmc_weighted_values for how these weights are calculated.
    """
    df = pd.merge(do_df, mcmc_df, on=["run", "chain"])
    df.drop(columns=["chain", "run", "loglikelihood", "ap_loglikelihood", "accept"], inplace=True)
    times = sorted(df["times"].unique())
    scenarios = df["scenario"].unique()
    uncertainty_data = []
    for scenario in scenarios:
        scenario_mask = df["scenario"] == scenario
        scenario_df = df[scenario_mask]
        for time in times:
            time_mask = scenario_df["times"] == time
            masked_df = scenario_df[time_mask]
            if masked_df.empty:
                continue

            for target in targets.values():
                quantiles = target["quantiles"]
                output_name = target["output_key"]
                weighted_values = np.repeat(masked_df[output_name], masked_df["weight"])
                quantile_vals = np.quantile(weighted_values, quantiles)
                for q_idx, q_value in enumerate(quantile_vals):
                    datum = {
                        "scenario": scenario,
                        "type": output_name,
                        "time": time,
                        "quantile": quantiles[q_idx],
                        "value": q_value,
                    }
                    uncertainty_data.append(datum)

    uncertainty_df = pd.DataFrame(uncertainty_data)
    return uncertainty_df
