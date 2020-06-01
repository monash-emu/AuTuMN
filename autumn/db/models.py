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


def collate_outputs_powerbi(
    src_db_paths: List[str], target_db_path: str, max_size: float
):
    """
    Collate the output of many calibration databases into a single database,
    of size no greater than `max_size` MB, then converts it into the PowerBI format.
    """
    # Figure out the number of runs to sample - we assume all source dbs are approximately the same size.
    sum_size_mb = 0
    sum_runs = 0
    with Timer("Measuring input databases"):
        for db_path in src_db_paths:
            db = Database(db_path)
            sum_size_mb += db.get_size_mb()
            sum_runs += db.engine.execute("SELECT COUNT(*) from mcmc_run;").first()[0]

        mb_per_run = sum_size_mb / sum_runs
        max_runs = max_size / mb_per_run

    # Collate the outputs
    with TemporaryDirectory() as temp_dir:
        # Figure out how much bigger the db gets due to pivoting
        with Timer("Calculating pivot factor"):
            sample_db_path = src_db_paths[0]
            # TODO: subsample just one run to make this faster.
            pivot_db_path = os.path.join(temp_dir, "pivot.db")
            create_power_bi_outputs(sample_db_path, pivot_db_path)
            sample_db_size = Database(sample_db_path).get_size_mb()
            pivot_db_size = Database(pivot_db_path).get_size_mb()
            pivot_factor = math.ceil(pivot_db_size / sample_db_size)
            # Throw away 20% as a safety factor
            num_runs = math.floor((4 / 5) * max_runs / len(src_db_paths) / pivot_factor)

        logger.info(
            "Sampling %s runs to achieve no more than %s MB", num_runs, max_size
        )
        collated_db_path = os.path.join(temp_dir, "collated.db")
        with Timer("Collating outputs into one database"):
            collate_outputs(src_db_paths, collated_db_path, num_runs)
        with Timer("Converting collated database to PowerBI format"):
            create_power_bi_outputs(collated_db_path, target_db_path)


def collate_outputs(src_db_paths: List[str], target_db_path: str, num_runs: int):
    """
    Collate the output of many calibration databases into a single database.
    Selects the top `num_runs` most recent accepted runs from each database.
    Run names are renamed to be ascending in the final database.
    """
    logger.info("Collating db outputs into %s", target_db_path)
    target_db = Database(target_db_path)
    run_count = 0
    for db_path in tqdm(src_db_paths):
        accepted_runs = {}
        source_db = Database(db_path)
        # Get a list of source run names
        mcmc_run_df = source_db.db_query("mcmc_run")
        accepted_df = mcmc_run_df[mcmc_run_df["accept"] == 1]
        num_accepted_runs = len(accepted_df)
        if num_accepted_runs >= num_runs:
            logger.warn(
                "Insufficient accepted runs in %s: %s requested but only %s available",
                db_path,
                num_runs,
                num_accepted_runs,
            )
        accepted_df = accepted_df.tail(num_runs)
        for _, row in accepted_df.iterrows():
            old_name = row.idx
            new_name = f"run_{run_count}"
            run_count += 1
            accepted_runs[old_name] = new_name

        tables_to_copy = ["outputs", "derived_outputs", "mcmc_run"]
        for table_name in tables_to_copy:
            table_df = source_db.db_query(table_name)
            mask = table_df["idx"] == None  # Init mask as all False
            for old_name, new_name in accepted_runs.items():
                mask |= table_df["idx"] == old_name

            table_df = table_df[mask]
            table_df = table_df.replace(accepted_runs)
            target_db.dump_df(table_name, table_df)

    logger.info("Finished collating db outputs into %s", target_db_path)


def create_power_bi_outputs(source_db_path: str, target_db_path: str):
    """
    Read the model outputs from a database and then convert them into a form
    that is readable by our PowerBI dashboard.
    Save the converted data into its own database.
    """
    source_db = Database(source_db_path)
    target_db = Database(target_db_path)
    tables_to_copy = ["mcmc_run", "derived_outputs"]
    for table_name in tables_to_copy:
        logger.info("Copying %s", table_name)
        table_df = source_db.db_query(table_name)
        target_db.dump_df(table_name, table_df)

    outputs_df = source_db.db_query("outputs")
    scenario_names = outputs_df["Scenario"].unique()
    logger.info("Converting scenarios to PowerBI format")
    for scenario_name in tqdm(scenario_names):
        scenario_idx = int(scenario_name.split("_")[-1])
        table_name = f"pbi_scenario_{scenario_idx}"
        mask = outputs_df["Scenario"] == scenario_name
        scenario_df = unpivot_outputs(outputs_df[mask])
        target_db.dump_df(table_name, scenario_df)

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
