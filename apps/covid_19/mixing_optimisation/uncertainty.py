import os
import logging
import multiprocessing
from concurrent import futures
from typing import List


import numpy as np
import pandas as pd


from autumn.db.database import Database


logger = logging.getLogger(__name__)


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
