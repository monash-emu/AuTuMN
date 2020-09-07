import os
import logging
import multiprocessing
from concurrent import futures
from typing import List


import numpy as np
import pandas as pd


from autumn.db.database import Database


logger = logging.getLogger(__name__)


def add_uncertainty_weights(output_names: List[str], database_path: str):
    """
    Calculate uncertainty weights for a given MCMC chain and derived output.
    Saves requested weights in a table 'uncertainty_weights'.
    """
    db = Database(database_path)

    # Delete old data.
    for output_name in output_names:
        if "uncertainty_weights" in db.table_names():
            logger.info(
                "Deleting %s from existing uncertainty_weights table in %s",
                output_name,
                database_path,
            )
            db.engine.execute(f"DELETE FROM uncertainty_weights WHERE output_name='{output_name}'")

    # Add new data
    logger.info("Loading MCMC run metadata into memory")
    mcmc_df = db.query("mcmc_run")

    logger.info("Loading derived outputs data into memory")
    columns = ["idx", "Scenario", "times", *output_names]
    derived_outputs_df = db.query("derived_outputs", column=columns)

    for output_name in output_names:
        logger.info("Adding uncertainty_weights for %s to %s", output_name, database_path)
        if "uncertainty_weights" in db.table_names():
            logger.info(
                "Deleting %s from existing uncertainty_weights table in %s",
                output_name,
                database_path,
            )
            db.engine.execute(f"DELETE FROM uncertainty_weights WHERE output_name='{output_name}'")

        weights_df = calculate_uncertainty_weights(output_name, mcmc_df, derived_outputs_df)
        db.dump_df("uncertainty_weights", weights_df)
        logger.info("Finished writing %s uncertainty weights", output_name)

    logger.info("Finished writing all uncertainty weights")


def calculate_uncertainty_weights(
    output_name: str, mcmc_df: pd.DataFrame, derived_outputs_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate uncertainty weights for a given MCMC chain and derived output.
    """
    logger.info("Calculating weighted values for %s", output_name)
    mcmc_df["run_int"] = mcmc_df["idx"].apply(run_idx_to_int)
    mcmc_df = mcmc_df.sort_values(["run_int"])
    max_run_idx = mcmc_df["run_int"].max()
    weights = np.zeros(max_run_idx + 1)
    last_run_int = None
    for run_int, accept in zip(mcmc_df["run_int"], mcmc_df["accept"]):
        if accept:
            weights[run_int] = 1
            last_run_int = run_int
        elif last_run_int is None:
            continue
        else:
            weights[last_run_int] += 1

    # Add weights to weights dataframe
    cols = ["idx", "Scenario", "times", output_name]
    weights_df = (
        derived_outputs_df.copy()
        .drop(columns=[c for c in derived_outputs_df.columns if c not in cols])
        .rename(columns={output_name: "value"})
    )
    weights_df["output_name"] = output_name
    select_weight = lambda run_idx: weights[run_idx_to_int(run_idx)]
    weights_df["weight"] = weights_df["idx"].apply(select_weight)
    return weights_df


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
    weights_df = db.query("uncertainty_weights")
    logger.info("Calculating uncertainty")
    uncertainty_df = calculate_mcmc_uncertainty(weights_df, targets)
    db.dump_df("uncertainty", uncertainty_df)
    logger.info("Finished writing uncertainties")


def calculate_mcmc_uncertainty(weights_df: pd.DataFrame, targets: dict) -> pd.DataFrame:
    """
    Calculate quantiles from a table of weighted values.
    See calc_mcmc_weighted_values for how these weights are calculated.
    """
    times = sorted(weights_df.times.unique())
    scenarios = weights_df.Scenario.unique()
    uncertainty_data = []
    for scenario in scenarios:
        scenario_mask = weights_df["Scenario"] == scenario
        scenario_df = weights_df[scenario_mask]
        for target in targets.values():
            quantiles = target["quantiles"]
            output_name = target["output_key"]

            output_mask = scenario_df["output_name"] == output_name
            output_df = scenario_df[output_mask]
            for time in times:
                time_mask = output_df["times"] == time
                masked_df = output_df[time_mask]
                if masked_df.empty:
                    continue

                weighted_values = np.repeat(masked_df.value, masked_df.weight)
                quantile_vals = np.quantile(weighted_values, quantiles)
                for q_idx, q_value in enumerate(quantile_vals):
                    datum = {
                        "Scenario": scenario,
                        "type": output_name,
                        "time": time,
                        "quantile": quantiles[q_idx],
                        "value": q_value,
                    }
                    uncertainty_data.append(datum)

    uncertainty_df = pd.DataFrame(uncertainty_data)
    return uncertainty_df


def run_idx_to_int(run_idx: str) -> int:
    return int(run_idx.split("_")[-1])


def run_int_to_idx(run_int: int) -> str:
    return f"run_{run_int}"


def collect_all_mcmc_output_tables(calib_dirpath):
    db_paths = [
        os.path.join(calib_dirpath, f)
        for f in os.listdir(calib_dirpath)
        if f.endswith(".db") and not f.startswith("mcmc_percentiles")
    ]
    mcmc_tables = []
    output_tables = []
    derived_output_tables = []
    for db_path in db_paths:
        db = Database(db_path)
        mcmc_tables.append(db.query("mcmc_run"))
        output_tables.append(db.query("outputs"))
        derived_output_tables.append(db.query("derived_outputs"))
    return mcmc_tables, output_tables, derived_output_tables


def collect_iteration_weights(mcmc_tables: List[pd.DataFrame], burn_in=0):
    """
    Work out the weights associated with accepted iterations, considering how many rejections followed each acceptance
    :param mcmc_tables: list of mcmc output tables
    :param burn_in: number of discarded iterations
    :return: list of dictionaries (one dictionary per MCMC chain)
        keys are the run ids and values are the iteration weights
    """
    weights = []
    for i_chain in range(len(mcmc_tables)):
        mcmc_tables[i_chain].sort_values(["idx"])
        weight_dict = {}
        last_run_id = None
        for i_row, run_id in enumerate(mcmc_tables[i_chain].idx):
            if int(run_id[4:]) < burn_in:
                continue
            if mcmc_tables[i_chain].accept[i_row] == 1:
                weight_dict[run_id] = 1
                last_run_id = run_id
            elif last_run_id is None:
                continue
            else:
                weight_dict[last_run_id] += 1
        weights.append(weight_dict)
    return weights


def export_compartment_size(
    compartment_name, mcmc_tables, output_tables, derived_output_tables, weights, scenario="S_0"
):
    if "start_time" in mcmc_tables[0].columns:
        # Find the earliest time that is common to all accepted runs (if start_time was varied).
        max_start_time = 0
        for mcmc_table_df in mcmc_tables:
            mask = mcmc_table_df["accept"] == 1
            _max_start_time = mcmc_table_df[mask]["start_time"].max()
            if _max_start_time > max_start_time:
                max_start_time = _max_start_time
        t_min = round(max_start_time)
    mask = output_tables[0]["Scenario"] == scenario
    times = [t for t in output_tables[0][mask]["times"].unique() if t >= t_min]

    compartment_values = {}
    for i_time, time in enumerate(times):
        output_list = []
        for i_chain in range(len(mcmc_tables)):
            for run_id, w in weights[i_chain].items():
                mask = (
                    (output_tables[i_chain].idx == run_id)
                    & (output_tables[i_chain].times == time)
                    & (output_tables[i_chain].Scenario == scenario)
                )
                output_val = float(output_tables[i_chain][compartment_name][mask])
                output_list += [output_val] * w
        compartment_values[str(time)] = output_list

    return compartment_values
