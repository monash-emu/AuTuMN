"""
Processing data from the output database.
"""
import logging
from typing import List

import numpy as np
import pandas as pd
import random

from autumn.db.database import BaseDatabase, get_database
from autumn.db.load import load_mcmc_tables
from autumn.utils.params import load_params, load_targets
from utils.runs import read_run_id

logger = logging.getLogger(__name__)


def collate_databases(src_db_paths: List[str], target_db_path: str, tables=None):
    """
    Collate the output of many calibration databases into a single database.
    Run names are renamed to be ascending in the final database.
    """
    logger.info("Collating db outputs into %s", target_db_path)
    target_db = get_database(target_db_path)
    for db_path in src_db_paths:
        logger.info("Reading data from %s", db_path)
        source_db = get_database(db_path)
        for table_name in source_db.table_names():
            if tables and table_name not in tables:
                logger.info("Skipping table %s", table_name)
                continue

            logger.info("Copying table %s", table_name)
            table_df = source_db.query(table_name)
            target_db.dump_df(table_name, table_df)

    logger.info("Finished collating db outputs into %s", target_db_path)


def find_mle_run(df: pd.DataFrame) -> pd.DataFrame:
    accept_mask = df["accept"] == 1
    max_ll = df[accept_mask]["loglikelihood"].max()
    max_ll_mask = accept_mask & (df["loglikelihood"] == max_ll)
    return df[max_ll_mask].copy()


def find_mle_params(mcmc_df: pd.DataFrame, param_df: pd.DataFrame) -> dict:
    mle_run_df = find_mle_run(mcmc_df)
    run_id = mle_run_df["run"].iloc[0]
    chain_id = mle_run_df["chain"].iloc[0]
    param_mask = (param_df["run"] == run_id) & (param_df["chain"] == chain_id)
    params = {}
    for _, row in param_df[param_mask].iterrows():
        params[row["name"]] = row["value"]

    return params


def select_pruning_candidates(src_db_path: str, n_candidates: int) -> pd.DataFrame:
    """Select a random set of 'good enough' candidates for manual inspection
    The output set will be guaranteed to contain the highest
    MLE run from all the chains, in addition to randomly selected candidates

    Args:
        src_db_path (str): Base path of calibration run (containing subdirectories for each chain)
        n_candidates (int): Number of candidates to select

    Returns:
        candidates (pd.DataFrame): DataFrame containing unique identifiers (chain_id, run_id) of all candidates

    """

    #+++ FIXME/TODO
    # We just use a naive random selection, disregarding burn-in etc
    # Could possibly use selection routine from sample_outputs_for_calibration_fit

    # Load all MCMC run data to select from

    all_mcmc_df = pd.concat(load_mcmc_tables(src_db_path), ignore_index=True)

    all_accepted = all_mcmc_df[all_mcmc_df["accept"]==1]

    # Find the MLE candidate
    max_ll = all_accepted["loglikelihood"].max()
    max_ll_candidate = all_accepted[all_accepted["loglikelihood"] == max_ll].iloc[0].name

    # Sample random candidates
    possible_candidates = list(all_accepted.index)
    possible_candidates.remove(max_ll_candidate)

    candidates = random.sample(possible_candidates, k = n_candidates-1)
    candidates.append(max_ll_candidate)

    candidates_df = all_accepted.loc[candidates]

    return candidates_df


def prune_chain(source_db_path: str, target_db_path: str):
    """
    Read the model outputs from a database and removes output data that is not MLE.
    This is an operation applied to each chain's database.
    """
    logger.info("Pruning %s into %s", source_db_path, target_db_path)
    source_db = get_database(source_db_path)
    target_db = get_database(target_db_path)

    # Find the maximum accepted loglikelihood for all runs
    mcmc_run_df = source_db.query("mcmc_run")
    mle_run_df = find_mle_run(mcmc_run_df)
    mle_run_id = mle_run_df.run.iloc[0]
    mle_chain_id = mle_run_df.chain.iloc[0]
    # Copy tables over, pruning some.
    tables_to_copy = source_db.table_names()
    for table_name in tables_to_copy:
        table_df = source_db.query(table_name)
        if table_name == "outputs":
            # Drop everything except the MLE run
            logger.info("Pruning outputs so that it only contains max likelihood runs")
            mle_mask = (table_df["run"] == mle_run_id) & (table_df["chain"] == mle_chain_id)
            max_ll_table_df = table_df[mle_mask]
            target_db.dump_df(table_name, max_ll_table_df)
        elif table_name:
            # Copy table over (mcmc_run, mcmc_params, derived_outputs)
            # We need to keep derived outputs here to be used by uncertainty calculations
            logger.info("Copying %s", table_name)
            target_db.dump_df(table_name, table_df)

    logger.info("Finished pruning %s into %s", source_db_path, target_db_path)


def prune_final(source_db_path: str, target_db_path: str):
    """
    Read the model outputs from a database and remove all run-related data that is not MLE.
    This is the final pruning for the collated database.
    """
    logger.info("Pruning %s into %s", source_db_path, target_db_path)
    source_db = get_database(source_db_path)
    target_db = get_database(target_db_path)

    # Find the maximum accepted loglikelihood for all runs
    mcmc_run_df = source_db.query("mcmc_run")
    mle_run_df = find_mle_run(mcmc_run_df)
    mle_run_id = mle_run_df.run.iloc[0]
    mle_chain_id = mle_run_df.chain.iloc[0]
    # Copy tables over, pruning some.
    tables_to_copy = source_db.table_names()
    for table_name in tables_to_copy:
        table_df = source_db.query(table_name)
        if table_name == "outputs":
            # Drop everything except the MLE run
            logger.info("Pruning outputs so that it only contains max likelihood runs")
            mle_mask = (table_df["run"] == mle_run_id) & (table_df["chain"] == mle_chain_id)
            max_ll_table_df = table_df[mle_mask]
            target_db.dump_df(table_name, max_ll_table_df)
        elif table_name == "derived_outputs":
            # Drop everything except the MLE run
            logger.info("Pruning derived_outputs so that it only contains max likelihood runs")
            mle_mask = (table_df["run"] == mle_run_id) & (table_df["chain"] == mle_chain_id)
            max_ll_table_df = table_df[mle_mask]
            target_db.dump_df(table_name, max_ll_table_df)
        elif table_name:
            # Copy table over (mcmc_run, mcmc_params)
            logger.info("Copying %s", table_name)
            target_db.dump_df(table_name, table_df)

    logger.info("Finished pruning %s into %s", source_db_path, target_db_path)


def powerbi_postprocess(source_db_path: str, target_db_path: str, run_id: str):
    """
    Read the model outputs from a database and then convert them into a form
    that is readable by our PowerBI dashboard.
    Save the converted data into its own database.
    """
    source_db = get_database(source_db_path)
    target_db = get_database(target_db_path)
    tables_to_copy = [t for t in source_db.table_names() if t != "outputs"]
    for table_name in tables_to_copy:
        logger.info("Copying %s", table_name)
        table_df = source_db.query(table_name)
        if table_name == "uncertainty":
            # Rename "time" field to "times"
            table_df.rename(columns={"time": "times"})

        target_db.dump_df(table_name, table_df)

    app_name, region_name, timestamp, git_commit = read_run_id(run_id)

    # Add build metadata table
    build_key = f"{timestamp}-{git_commit}"
    logger.info("Adding 'build' metadata table with key %s", build_key)
    build_df = pd.DataFrame.from_dict(
        {"build_key": [build_key], "app_name": [app_name], "region_name": [region_name]}
    )
    target_db.dump_df("build", build_df)

    # Add scenario metadata table
    logger.info("Adding 'scenario' metadata table")
    params = load_params(app_name, region_name)
    # Add default scenario
    scenario_data = [
        {
            "scenario": 0,
            "start_time": int(params["default"]["time"]["start"]),
            "description": params["default"].get("description", ""),
        }
    ]
    for sc_idx, sc_params in params["scenarios"].items():
        sc_datum = {
            "scenario": int(sc_idx),
            "start_time": int(sc_params["time"]["start"]),
            "description": sc_params.get("description", ""),
        }
        scenario_data.append(sc_datum)

    scenario_df = pd.DataFrame(scenario_data)
    target_db.dump_df("scenario", scenario_df)

    # Add calibration targets
    logger.info("Adding 'targets' table")
    targets = load_targets(app_name, region_name)
    targets_data = []
    for target in targets.values():
        for t, v in zip(target["times"], target["values"]):
            t_datum = {
                "key": target["output_key"],
                "times": t,
                "value": v,
            }
            targets_data.append(t_datum)

    targets_df = pd.DataFrame(targets_data)
    target_db.dump_df("targets", targets_df)

    logger.info("Converting outputs to PowerBI format")
    outputs_df = source_db.query("outputs")
    pbi_outputs_df = unpivot_outputs(outputs_df)
    target_db.dump_df("powerbi_outputs", pbi_outputs_df)
    logger.info("Finished creating PowerBI output database at %s", target_db_path)


def unpivot_outputs(output_df: pd.DataFrame):
    """
    Take outputs in the form they come out of the model object and convert them into a "long", "melted" or "unpiovted"
    format in order to more easily plug to PowerBI
    """
    id_cols = ["chain", "run", "scenario", "times"]
    value_cols = [c for c in output_df.columns if c not in id_cols]
    output_df = output_df.melt(id_vars=id_cols, value_vars=value_cols)
    cols = {"compartment"}

    def label_strata(row: list):
        strata = {"compartment": row[0]}
        for el in row[1:]:
            parts = el.split("_")
            k = parts[0]
            # FIXME: Use this once Milinda can use it in PowerBI
            # v = "_".join(parts[1:])
            strata[k] = el
            cols.add(k)

        return strata

    variables = (s.split("X") for s in output_df.variable)
    new_cols_df = pd.DataFrame([label_strata(row) for row in variables])
    output_df = output_df.join(new_cols_df)
    output_df = output_df.drop(columns="variable")
    return output_df


def sample_runs(mcmc_df: pd.DataFrame, num_samples: int):
    """
    Returns a list of chain ids + run ids for each sampled run.
    Choose runs with probability proprotional to their acceptance weights.
    """
    run_choices = list(zip(mcmc_df["chain"].tolist(), mcmc_df["run"].tolist()))
    assert num_samples < len(run_choices), "Must be more samples than choices"
    weights = mcmc_df["weight"].to_numpy()
    sample_pr = weights / weights.sum()
    idxs = np.array([i for i in range(len(weights))])
    chosen_idxs = np.random.choice(idxs, size=num_samples, replace=False, p=sample_pr)
    chosen_runs = [run_choices[i] for i in chosen_idxs]
    return chosen_runs
