"""
Processing data from the output database.
"""
import logging
from typing import List
from datetime import date

import numpy as np
import pandas as pd

from autumn.core.db.database import get_database
from autumn.core.db.load import load_mcmc_tables
from autumn.core.utils.runs import read_run_id

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


def get_identifying_run_ids(table: pd.DataFrame) -> pd.Series:
    """
    Args:
        table (pd.DataFrame): Table with 'run' and 'chain' columns

    Returns:
        pd.Series: Combined run identifier of same length as table

    """
    return table["chain"].astype(str) + ":" + table["run"].astype(str)


def select_pruning_candidates(sampled_runs_df: pd.DataFrame, n_candidates: int, weighted=True) -> pd.DataFrame:
    """Select a random set of 'good enough' candidates for manual inspection
    The output set will be guaranteed to contain the highest
    MLE run from all the chains, in addition to randomly selected candidates

    Args:
        src_db_path (str): Base path of calibration run (containing subdirectories for each chain)
        n_candidates (int): Number of candidates to select.  If 1, then only the MLE run from all chains will be selected
        weighted (bool): Weight candidates by 1.0/loglikelihood (False means uniform selection)
    Returns:
        candidates (pd.DataFrame): DataFrame containing unique identifiers (chain_id, run_id) of all candidates

    """

    # +++ FIXME/TODO
    # We just use a naive random selection, disregarding burn-in etc
    # Could possibly use selection routine from sample_outputs_for_calibration_fit

    # Load all MCMC run data to select from

    #all_accepted = all_mcmc_df[all_mcmc_df["accept"] == 1]

    # Find the MLE candidate
    max_ll = sampled_runs_df["loglikelihood"].max()
    max_ll_candidate = sampled_runs_df[sampled_runs_df["loglikelihood"] == max_ll].iloc[0].name

    # Ensure candidates have been sampled and that output data is available
    #accepted_and_sampled = all_accepted[all_accepted["sampled"] == 1]

    # Sample random candidates
    possible_candidates = list(sampled_runs_df.index)
    if max_ll_candidate in possible_candidates:
        possible_candidates.remove(max_ll_candidate)

    if weighted:
        # +++ FIXME Adding 10.0 to not overweight, should parameterise this
        weights = 1.0 / (
            10.0 + np.abs(np.array(sampled_runs_df.loc[possible_candidates].loglikelihood))
        )
        weights = weights / weights.sum()
    else:
        weights = None

    # Ensure we aren't sampling too many candidates (most likely to show up in testing)
    n_candidates = min(n_candidates, len(possible_candidates))

    candidates = list(
        np.random.choice(possible_candidates, n_candidates - 1, replace=False, p=weights)
    )

    # Ensure we have the max likelihood candidate

    candidates.append(max_ll_candidate)

    candidates_df = sampled_runs_df.loc[candidates]

    return candidates_df


def prune_chain(source_db_path: str, target_db_path: str, chain_candidates: pd.DataFrame):
    """
    Read the model outputs from a database and removes output data that is not MLE.
    This is an operation applied to each chain's database.
    """
    logger.info("Pruning %s into %s", source_db_path, target_db_path)
    source_db = get_database(source_db_path)
    target_db = get_database(target_db_path)

    # Copy tables over, pruning some.
    tables_to_copy = source_db.table_names()
    for table_name in tables_to_copy:
        table_df = source_db.query(table_name)
        if table_name == "outputs":
            # Drop everything except the MLE run
            logger.info("Pruning outputs so that it only contains candidate runs")
            candidate_mask = table_df["run"].isin(chain_candidates["run"])
            candidate_table_df = table_df[candidate_mask]
            target_db.dump_df(table_name, candidate_table_df)
        elif table_name:
            # Copy table over (mcmc_run, mcmc_params, derived_outputs)
            # We need to keep derived outputs here to be used by uncertainty calculations
            logger.info("Copying %s", table_name)
            target_db.dump_df(table_name, table_df)

    logger.info("Finished pruning %s into %s", source_db_path, target_db_path)


def prune_final(source_db_path: str, target_db_path: str, candidates_df: pd.DataFrame):
    """
    Read the model outputs from a database and remove all run-related data that is not MLE.
    This is the final pruning for the collated database.
    """
    logger.info("Pruning %s into %s", source_db_path, target_db_path)
    source_db = get_database(source_db_path)
    target_db = get_database(target_db_path)

    # Copy tables over, pruning some.
    tables_to_copy = source_db.table_names()
    for table_name in tables_to_copy:
        table_df = source_db.query(table_name)
        if table_name == "derived_outputs":
            # Drop everything except the candidate runs
            logger.info("Pruning derived_outputs so that it only contains candidate runs")
            candidate_iruns = get_identifying_run_ids(candidates_df)
            table_df["irun_id"] = get_identifying_run_ids(table_df)
            filtered_table_df = table_df[table_df["irun_id"].isin(candidate_iruns)]
            final_df = filtered_table_df.drop(columns="irun_id")
            target_db.dump_df(table_name, final_df)
        elif table_name:
            # Copy table over (outputs, mcmc_run, mcmc_params)
            # Note: Outputs has already been pruned to candidates in early prune_chains sweep
            logger.info("Copying %s", table_name)
            target_db.dump_df(table_name, table_df)

    logger.info("Finished pruning %s into %s", source_db_path, target_db_path)


def powerbi_postprocess(source_db_path: str, target_db_path: str, run_id: str):
    """
    Read the model outputs from a database and then convert them into a form
    that is readable by our PowerBI dashboard.
    Save the converted data into its own database.
    """
    from autumn.core.project import get_project

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

    project = get_project(app_name, region_name)
    basline_params = project.param_set.baseline.to_dict()
    sc_params = [sc.to_dict() for sc in project.param_set.scenarios]

    # Add default scenario
    scenario_data = [
        {
            "scenario": 0,
            "start_time": int(basline_params["time"]["start"]),
            "description": basline_params.get("description", ""),
        }
    ]
    for sc_idx, sc_params in enumerate(sc_params):
        sc_datum = {
            "scenario": int(sc_idx + 1),
            "start_time": int(sc_params["time"]["start"]),
            "description": sc_params.get("description", ""),
        }
        scenario_data.append(sc_datum)

    scenario_df = pd.DataFrame(scenario_data)
    target_db.dump_df("scenario", scenario_df)

    # Add calibration targets
    logger.info("Adding 'targets' table")
    targets_data = []
    for target in project.calibration.targets:
         targets_data += [{'key': target.data.name, 'times': idx, 'value': v} for idx,v in target.data.iteritems()]

    targets_df = pd.DataFrame(targets_data)
    target_db.dump_df("targets", targets_df)

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


def select_outputs_from_candidates(
    output_name: str,
    derived_output_tables: pd.DataFrame,
    candidates_df: pd.DataFrame,
    ref_date: date,
):
    out_df = pd.DataFrame()
    for idx, c in candidates_df.iterrows():
        chain = int(c["chain"])
        run = int(c["run"])
        ctable = derived_output_tables[chain]
        run_mask = ctable["run"] == run
        scenario_mask = ctable["scenario"] == 0
        masked = ctable[run_mask & scenario_mask]
        name = f"{chain}_{run}"
        out_df[name] = pd.Series(
            index=timelist_to_dti(masked["times"], ref_date), data=masked[output_name].data
        )
    return out_df


def timelist_to_dti(times, ref_date):
    datelist = [ref_date + pd.offsets.Day(t) for t in times]
    return pd.DatetimeIndex(datelist)


def target_to_series(target, ref_date):
    index = timelist_to_dti(target["times"], ref_date)
    return pd.Series(index=index, data=target["values"])
