"""
Loading data from the output database.
"""
import os
import logging
from typing import List

import numpy
import pandas as pd

from summer.model import StratifiedModel

from ..db.database import Database
from autumn.tb_model.loaded_model import LoadedModel
from autumn.tool_kit import Scenario


logger = logging.getLogger(__name__)


def load_model_scenarios(database_path: str, model_params: dict) -> List[Scenario]:
    """
    Load model scenarios from an output database.
    Will apply post processing if the post processing config is supplied.
    Will store model params in the database if suppied.
    """
    out_db = Database(database_path=database_path)
    scenarios = []
    # Load runs from the database
    run_results = out_db.engine.execute("SELECT DISTINCT run FROM outputs;").fetchall()
    run_names = sorted(([result[0] for result in run_results]))
    for run_name in run_names:
        # Load scenarios from the database for this run
        scenario_results = out_db.engine.execute(
            f"SELECT DISTINCT scenario FROM outputs WHERE run='{run_name}';"
        ).fetchall()
        scenario_names = sorted(([result[0] for result in scenario_results]))
        for scenario_name in scenario_names:
            # Load model outputs from database, build Scenario instance
            conditions = [f"scenario='{scenario_name}'", f"run='{run_name}'"]
            outputs = out_db.query("outputs", conditions=conditions)
            derived_outputs = out_db.query("derived_outputs", conditions=conditions)
            model = LoadedModel(
                outputs=outputs.to_dict(), derived_outputs=derived_outputs.to_dict()
            )
            scenario = Scenario.load_from_db(scenario_name, model, params=model_params)
            scenarios.append(scenario)

    return scenarios


def load_mcmc_tables(calib_dirpath: str):
    mcmc_tables = []
    for db_path in _find_db_paths(calib_dirpath):
        db = Database(db_path)
        mcmc_tables.append(db.query("mcmc_run"))

    return mcmc_tables


def load_mcmc_params(db: Database, run_id: int):
    """
    Returns a dict of params
    """
    params_df = db.query("mcmc_params", conditions=[f"run={run_id}"])
    return {row["name"]: row["value"] for _, row in params_df.iterrows()}


def load_mcmc_params_tables(calib_dirpath: str):
    mcmc_tables = []
    for db_path in _find_db_paths(calib_dirpath):
        db = Database(db_path)
        mcmc_tables.append(db.query("mcmc_params"))

    return mcmc_tables


def load_derived_output_tables(calib_dirpath: str, column: str = None):
    derived_output_tables = []
    for db_path in _find_db_paths(calib_dirpath):
        db = Database(db_path)
        if not column:
            df = db.query("derived_outputs")
            derived_output_tables.append(df)
        else:
            cols = ["chain", "run", "scenario", "times", column]
            df = db.query("derived_outputs", column=cols)
            derived_output_tables.append(df)

    return derived_output_tables


def _find_db_paths(dirpath: str):
    db_paths = [
        os.path.join(dirpath, f)
        for f in os.listdir(dirpath)
        if f.endswith(".db") and not f.startswith("mcmc_percentiles")
    ]
    return sorted(db_paths)
