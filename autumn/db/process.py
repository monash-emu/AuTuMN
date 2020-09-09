"""
Processing data from the output database.
"""
import os
import logging
from typing import List

import pandas as pd

from ..db.database import Database


logger = logging.getLogger(__name__)


def apply_burn_in(src_db: Database, dest_db: Database, burn_in: int):
    """
    Copy mcmc_run and mcmc_params table from src_db to dest_db with a burn in applied.
    """
    logger.info(
        "Copying burnt-in mcmc_run and mcmc_params tables from %s to  %s",
        src_db.database_path,
        dest_db.database_path,
    )

    mcmc_run_df = src_db.query("mcmc_run")
    num_runs = len(mcmc_run_df)
    assert num_runs > burn_in, f"Tried to burn {burn_in} runs, but there are only {num_runs}"
    burn_mask = mcmc_run_df["run"] >= burn_in
    mcmc_run_burned_df = mcmc_run_df[burn_mask]
    burned_runs_str = ", ".join([str(i) for i in mcmc_run_df[~burn_mask].run])
    logger.info("Burned MCMC runs %s", burned_runs_str)

    mcmc_params_df = src_db.query("mcmc_params")
    burn_mask = mcmc_params_df["run"] >= burn_in
    mcmc_params_burned_df = mcmc_params_df[burn_mask]

    dest_db.dump_df("mcmc_run", mcmc_run_burned_df)
    dest_db.dump_df("mcmc_params", mcmc_params_burned_df)


def collate_databases(src_db_paths: List[str], target_db_path: str):
    """
    Collate the output of many calibration databases into a single database.
    Run names are renamed to be ascending in the final database.
    """
    logger.info("Collating db outputs into %s", target_db_path)
    target_db = Database(target_db_path)
    for db_path in src_db_paths:
        source_db = Database(db_path)
        for table_name in source_db.table_names():
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


def prune(source_db_path: str, target_db_path: str, targets=None):
    """
    Read the model outputs from a database and remove all run-related data that is not MLE.
    """
    logger.info("Pruning %s into %s", source_db_path, target_db_path)
    source_db = Database(source_db_path)
    target_db = Database(target_db_path)

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
        elif table_name == "derived_outputs" and targets:
            # Drop all columns that aren't targets
            output_keys = [t["output_key"] for t in targets.values()]
            logger.info(
                "Pruning derived_outputs so that it only contains target outputs: %s", output_keys
            )
            cols = ["chain", "run", "scenario", "times", *output_keys]
            drop_cols = [c for c in table_df.columns if c not in cols]
            table_df.drop(columns=drop_cols, inplace=True)
            target_db.dump_df(table_name, table_df)
        elif table_name == "derived_outputs":
            # Drop everything except the MLE run
            logger.info("Pruning derived_outputs so that it only contains max likelihood runs")
            mle_mask = (table_df["run"] == mle_run_id) & (table_df["chain"] == mle_chain_id)
            max_ll_table_df = table_df[mle_mask]
            target_db.dump_df(table_name, max_ll_table_df)
        elif table_name:
            # Copy table over
            logger.info("Copying %s", table_name)
            target_db.dump_df(table_name, table_df)

    logger.info("Finished pruning %s into %s", source_db_path, target_db_path)


def unpivot(source_db_path: str, target_db_path: str):
    """
    Read the model outputs from a database and then convert them into a form
    that is readable by our PowerBI dashboard.
    Save the converted data into its own database.
    """
    source_db = Database(source_db_path)
    target_db = Database(target_db_path)
    tables_to_copy = [t for t in source_db.table_names() if t != "outputs"]
    for table_name in tables_to_copy:
        logger.info("Copying %s", table_name)
        table_df = source_db.query(table_name)
        target_db.dump_df(table_name, table_df)

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
