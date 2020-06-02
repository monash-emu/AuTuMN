import os
import logging
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from autumn.db.database import Database

DEFAULT_QUANTILES = [0.025, 0.25, 0.5, 0.75, 0.975]

logger = logging.getLogger(__file__)


def add_uncertainty(output_name: str, database_path: str):
    """
    Add an uncertainty table to a given database, based on mcmc_run and derived_outputs.
    The table will have columns scenario/type/time/quantile/value.
    """
    logger.info("Adding uncertainty for %s to %s", output_name, database_path)
    db = Database(database_path)

    if "uncertainty" in db.table_names():
        logger.info(
            "Deleting %s from existing uncertainty table in %s",
            output_name,
            database_path,
        )
        db.engine.execute(f"DELETE FROM uncertainty WHERE type={output_name}")

    mcmc_tables = [db.db_query("mcmc_run")]
    output_tables = [db.db_query("outputs")]
    derived_output_tables = [db.db_query("derived_outputs")]
    logger.info("Calculating weights for %s", output_name)
    weights = collect_iteration_weights(mcmc_tables, burn_in=0)
    logger.info("Calculating uncertainty for %s", output_name)
    times_list, quantiles_list = compute_mcmc_output_quantiles(
        mcmc_tables,
        output_tables,
        derived_output_tables,
        weights,
        output_name,
        DEFAULT_QUANTILES,
    )
    logger.info("Preparing scenarios for %s", output_name)
    num_scenarios = len(times_list)
    for idx in tqdm(range(num_scenarios)):
        scenario_name = f"S_{idx}"
        times = times_list[idx]
        quantiles = quantiles_list[idx]
        scenario_data = []
        for t_idx, time in enumerate(times):
            for q_idx, value in enumerate(quantiles[t_idx]):
                datum = {
                    "Scenario": scenario_name,
                    "type": output_name,
                    "time": time,
                    "quantile": DEFAULT_QUANTILES[q_idx],
                    "value": value,
                }
                scenario_data.append(datum)

        logger.info(
            "Writing %s uncertainties for scenario %s", output_name, scenario_name
        )
        scenario_df = pd.DataFrame(scenario_data)
        db.dump_df("uncertainty", scenario_df)

    logger.info("Finished writing %s uncertainties", output_name)


def export_mcmc_quantiles(
    path_to_mcmc_outputs, output_names: List[str], q_list=DEFAULT_QUANTILES, burn_in=0,
):
    """
    Create a separate database containing MCMC quartile data, based on a set of MCMC output databases
    """
    out_db_path = os.path.join(
        path_to_mcmc_outputs, "mcmc_percentiles_burned_" + str(burn_in) + ".db"
    )
    output_db = Database(out_db_path)
    mcmc_tables, output_tables, derived_output_tables = collect_all_mcmc_output_tables(
        path_to_mcmc_outputs
    )
    weights = collect_iteration_weights(mcmc_tables, burn_in)
    for output_name in output_names:
        times, quantiles = compute_mcmc_output_quantiles(
            mcmc_tables,
            output_tables,
            derived_output_tables,
            weights,
            output_name,
            q_list,
        )
        column_names = ["times", "Scenario"]
        for q in q_list:
            c_name = "q_" + str(100 * q).replace(".", "_")
            if c_name[-2:] == "_0":
                c_name = c_name[:-2]
            column_names.append(c_name)
        out_table = pd.DataFrame(columns=column_names)
        for sc_index in range(len(times)):
            this_sc_table = pd.DataFrame(columns=column_names)
            this_sc_table.times = times[sc_index]
            this_sc_table.Scenario = ["S_" + str(sc_index)] * len(times[sc_index])
            for i_quantile, q in enumerate(q_list):
                c_name = column_names[i_quantile + 2]
                this_sc_table[c_name] = quantiles[sc_index][:, i_quantile]
            out_table = pd.concat([out_table, this_sc_table])

        output_db.dump_df(output_name, out_table)


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
        mcmc_tables.append(db.db_query("mcmc_run"))
        output_tables.append(db.db_query("outputs"))
        derived_output_tables.append(db.db_query("derived_outputs"))
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


def compute_mcmc_output_quantiles(
    mcmc_tables: List[pd.DataFrame],
    output_tables: List[pd.DataFrame],
    derived_output_tables: List[pd.DataFrame],
    weights: List[int],
    output_name: str,
    q_list: List[float],
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
    else:
        # Find the common start time.
        t_min = output_tables[0].times[0]

    # Find the common end time.
    t_max = list(output_tables[0].times)[-1]

    # Add cumulate output if required (not sure what this means).
    should_add_cumulate = (
        output_name not in derived_output_tables[0].columns
        and output_name[:8] == "cumulate"
    )
    if should_add_cumulate:
        derived_output_tables = add_cumulate_output_to_derived_outputs(
            derived_output_tables, output_name[9:]
        )

    scenario_list = sorted(derived_output_tables[0].Scenario.unique())
    quantiles_by_sc = []
    times_by_sc = []
    for i_scenario, scenario in tqdm(enumerate(scenario_list)):
        if scenario != "S_0":
            t_min = (
                derived_output_tables[0]
                .times[derived_output_tables[0].Scenario == scenario]
                .iloc[0]
            )

        steps = int(t_max - t_min + 1)
        times = np.linspace(t_min, t_max, num=steps).tolist()
        quantiles = np.zeros((len(times), 5))

        for i_time, time in enumerate(times):
            output_list = []
            for i_chain in range(len(mcmc_tables)):
                for run_id, w in weights[i_chain].items():
                    mask = (
                        (derived_output_tables[i_chain].idx == run_id)
                        & (derived_output_tables[i_chain].times == time)
                        & (derived_output_tables[i_chain].Scenario == scenario)
                    )
                    output_val = float(
                        derived_output_tables[i_chain][output_name][mask]
                    )
                    output_list += [output_val] * w
            quantiles[i_time, :] = np.quantile(output_list, q_list)

        quantiles_by_sc.append(quantiles)
        times_by_sc.append(times)

    return times_by_sc, quantiles_by_sc


def add_cumulate_output_to_derived_outputs(
    derived_output_tables: List[pd.DataFrame], output_name: str
):
    """
    Add an extra column in the output database for the cumulate value of a given output.
    """
    assert output_name in derived_output_tables[0].columns
    for i_chain in range(len(derived_output_tables)):
        derived_output_tables[i_chain].sort_values(by=["idx", "Scenario", "times"])
        cum_sum = 0.0
        cum_sum_column = []
        last_sc_index = derived_output_tables[i_chain]["Scenario"].iloc[0]
        last_run_index = derived_output_tables[i_chain]["idx"].iloc[0]
        for i_row in range(len(derived_output_tables[i_chain].index)):
            this_sc_index = derived_output_tables[i_chain]["Scenario"].iloc[i_row]
            this_run_index = derived_output_tables[i_chain]["idx"].iloc[i_row]
            value = derived_output_tables[i_chain][output_name].iloc[i_row]
            if this_sc_index == last_sc_index and this_run_index == last_run_index:
                cum_sum += value
            else:
                cum_sum = value
                if this_sc_index != last_sc_index:
                    last_sc_index = this_sc_index
                if this_run_index != last_run_index:
                    last_run_index = this_run_index
            cum_sum_column.append(cum_sum)
        derived_output_tables[i_chain]["cumulate_" + output_name] = cum_sum_column
    return derived_output_tables
