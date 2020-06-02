"""
Storing/loading/converting data from the output database.
"""
import os
import logging
import math
from typing import List
from tempfile import TemporaryDirectory

import numpy
import pandas as pd
from tqdm import tqdm

from summer.model import StratifiedModel

from ..db.database import Database, get_sql_engine
from autumn.tb_model.loaded_model import LoadedModel
from autumn.tool_kit import Scenario, Timer
from autumn.post_processing.processor import post_process


logger = logging.getLogger(__file__)


def load_model_scenarios(
    database_path: str, model_params={}, post_processing_config=None
) -> List[Scenario]:
    """
    Load model scenarios from an output database.
    Will apply post processing if the post processing config is supplied.
    Will store model params in the database if suppied.
    """
    out_db = Database(database_name=database_path)
    scenarios = []

    # Load runs from the database
    run_results = out_db.engine.execute("SELECT DISTINCT idx FROM outputs;").fetchall()
    run_names = sorted(([result[0] for result in run_results]))
    for run_name in run_names:
        # Load scenarios from the database for this run
        scenario_results = out_db.engine.execute(
            f"SELECT DISTINCT Scenario FROM outputs WHERE idx='{run_name}';"
        ).fetchall()
        scenario_names = sorted(([result[0] for result in scenario_results]))
        for scenario_name in scenario_names:
            # Load model outputs from database, build Scenario instance
            conditions = [f"Scenario='{scenario_name}'", f"idx='{run_name}'"]
            outputs = out_db.db_query("outputs", conditions=conditions)
            derived_outputs = out_db.db_query("derived_outputs", conditions=conditions)
            model = LoadedModel(
                outputs=outputs.to_dict(), derived_outputs=derived_outputs.to_dict()
            )
            idx = int(scenario_name.split("_")[1])
            chain_idx = int(run_name.split("_")[1])
            scenario = Scenario.load_from_db(idx, chain_idx, model, params=model_params)
            if post_processing_config:
                scenario.generated_outputs = post_process(model, post_processing_config)

            scenarios.append(scenario)

    return scenarios


def store_database(
    outputs, database_name, table_name="outputs", scenario=0, run_idx=0, times=None,
):
    """
    store outputs from the model in sql database for use in producing outputs later
    """
    if times:
        outputs.insert(0, column="times", value=times)

    if table_name != "mcmc_run_info":
        outputs.insert(0, column="idx", value=f"run_{run_idx}")
        outputs.insert(1, column="Scenario", value=f"S_{scenario}")

    store_db = Database(database_name)
    store_db.dump_df(table_name, outputs)


def store_run_models(models: List[StratifiedModel], database_path: str):
    """
    Store models in the database.
    Assume that models are sorted in an order such that their index is their scenario idx.
    """
    target_db = Database(database_path)
    for idx, model in enumerate(models):
        output_df = pd.DataFrame(model.outputs, columns=model.compartment_names)
        derived_output_df = pd.DataFrame.from_dict(model.derived_outputs)
        store_database(
            derived_output_df,
            scenario=idx,
            table_name="derived_outputs",
            database_name=database_path,
        )
        store_database(
            output_df,
            scenario=idx,
            table_name="outputs",
            times=model.times,
            database_name=database_path,
        )


def collate_databases(src_db_paths: List[str], target_db_path: str):
    """
    Collate the output of many calibration databases into a single database.
    Run names are renamed to be ascending in the final database.
    """
    logger.info("Collating db outputs into %s", target_db_path)
    target_db = Database(target_db_path)
    tables_to_copy = ["outputs", "derived_outputs", "mcmc_run"]
    run_count = 0
    for db_path in tqdm(src_db_paths):
        source_db = Database(db_path)
        num_runs = len(source_db.db_query("mcmc_run", column="idx"))
        for table_name in tables_to_copy:
            table_df = source_db.db_query(table_name)

            def increment_run(idx: str):
                run_idx = int(idx.split("_")[-1])
                new_idx = run_count + run_idx
                return f"run_{new_idx}"

            table_df.idx = table_df.idx.apply(increment_run)
            target_db.dump_df(table_name, table_df)

        run_count += num_runs

    logger.info("Finished collating db outputs into %s", target_db_path)


def prune(source_db_path: str, target_db_path: str):
    """
    Read the model outputs from a database and remove all run-related data that is not MLE.
    """
    logger.info("Pruning %s into %s", source_db_path, target_db_path)

    source_db = Database(source_db_path)
    target_db = Database(target_db_path)

    # Find the maximum loglikelihood for all runs
    mcmc_run_df = source_db.db_query("mcmc_run")
    max_ll_idx = mcmc_run_df.loglikelihood.idxmax()
    max_ll_run_name = mcmc_run_df.idx.iloc[max_ll_idx]
    tables_to_copy = [t for t in source_db.table_names()]
    for table_name in tables_to_copy:
        table_df = source_db.db_query(table_name)
        # Prune any table with an idx column except for mcmc_run
        should_prune = "idx" in table_df.columns and table_name != "mcmc_run"
        if should_prune:
            logger.info(
                "Pruning %s so that it only contains max likelihood runs", table_name
            )
            max_ll_mask = table_df["idx"] == max_ll_run_name
            max_ll_table_df = table_df[max_ll_mask]
            target_db.dump_df(table_name, max_ll_table_df)
        else:
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
        table_df = source_db.db_query(table_name)
        target_db.dump_df(table_name, table_df)

    logger.info("Converting outputs to PowerBI format")
    outputs_df = source_db.db_query("outputs")
    pbi_outputs_df = unpivot_outputs(outputs_df)
    target_db.dump_df("powerbi_outputs", pbi_outputs_df)
    logger.info("Finished creating PowerBI output database at %s", target_db_path)


def unpivot_outputs(output_df: pd.DataFrame):
    """
    Take outputs in the form they come out of the model object and convert them into a "long", "melted" or "unpiovted"
    format in order to more easily plug to PowerBI
    """
    id_cols = ["idx", "Scenario", "times"]
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


def load_calibration_from_db(database_directory, n_burned_per_chain=0):
    """
    Load all model runs stored in multiple databases found in database_directory
    :param database_directory: path to databases location
    :return: list of models
    """
    # list all databases
    db_names = os.listdir(database_directory + "/")
    db_names = [s for s in db_names if s[-3:] == ".db"]

    models = []
    n_loaded_iter = 0
    for db_name in db_names:
        out_database = Database(database_name=database_directory + "/" + db_name)

        # find accepted run indices
        res = out_database.db_query(
            table_name="mcmc_run", column="idx", conditions=["accept=1"]
        )
        run_ids = list(res.to_dict()["idx"].values())
        # find weights to associate with the accepted runs
        accept = out_database.db_query(table_name="mcmc_run", column="accept")
        accept = accept["accept"].tolist()
        one_indices = [i for i, val in enumerate(accept) if val == 1]
        one_indices.append(len(accept))  # add extra index for counting
        weights = [
            one_indices[j + 1] - one_indices[j] for j in range(len(one_indices) - 1)
        ]

        # burn fist iterations
        cum_sum = numpy.cumsum(weights).tolist()
        if cum_sum[-1] <= n_burned_per_chain:
            continue
        retained_indices = [i for i, c in enumerate(cum_sum) if c > n_burned_per_chain]
        run_ids = run_ids[retained_indices[0] :]
        previous_cum_sum = (
            cum_sum[retained_indices[0] - 1] if retained_indices[0] > 0 else 0
        )
        weights[retained_indices[0]] = weights[retained_indices[0]] - (
            n_burned_per_chain - previous_cum_sum
        )
        weights = weights[retained_indices[0] :]
        n_loaded_iter += sum(weights)
        for i, run_id in enumerate(run_ids):
            outputs = out_database.db_query(
                table_name="outputs", conditions=["idx='" + str(run_id) + "'"]
            )
            output_dict = outputs.to_dict()

            if out_database.engine.dialect.has_table(
                out_database.engine, "derived_outputs"
            ):
                derived_outputs = out_database.db_query(
                    table_name="derived_outputs",
                    conditions=["idx='" + str(run_id) + "'"],
                )

                derived_outputs_dict = derived_outputs.to_dict()
            else:
                derived_outputs_dict = None
            model_info_dict = {
                "db_name": db_name,
                "run_id": run_id,
                "model": LoadedModel(output_dict, derived_outputs_dict),
                "weight": weights[i],
            }
            models.append(model_info_dict)

    print("MCMC runs loaded.")
    print("Number of loaded iterations after burn-in: " + str(n_loaded_iter))
    return models
