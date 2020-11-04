"""
Loading data from the output database.
"""
import os
import logging
from typing import List

import pandas as pd

from ..db.database import get_database, BaseDatabase
from autumn.tb_model.loaded_model import LoadedModel
from autumn.tool_kit import Scenario


logger = logging.getLogger(__name__)


def load_model_scenarios(database_path: str, model_params: dict) -> List[Scenario]:
    """
    Load model scenarios from an output database.
    Will apply post processing if the post processing config is supplied.
    Will store model params in the database if suppied.
    """
    out_db = get_database(database_path=database_path)
    scenarios = []
    out_df = out_db.query("outputs")
    do_df = out_db.query("derived_outputs")
    run_names = sorted(out_df["run"].unique().tolist())
    for run_name in run_names:
        # Load scenarios from the database for this run
        run_mask = out_df["run"] == run_name
        scenario_names = sorted(out_df[run_mask]["scenario"].unique().tolist())
        for scenario_name in scenario_names:
            # Load model outputs from database, build Scenario instance
            scenario_mask = (out_df["scenario"] == scenario_name) & run_mask
            outputs = out_df[scenario_mask].to_dict()
            derived_outputs = do_df[scenario_mask].to_dict()
            model = LoadedModel(outputs=outputs, derived_outputs=derived_outputs)
            scenario = Scenario.load_from_db(scenario_name, model, params=model_params)
            scenarios.append(scenario)

    return scenarios


def load_mcmc_params(db: BaseDatabase, run_id: int):
    """
    Returns a dict of params
    """
    params_df = db.query("mcmc_params", conditions={"run": run_id})
    return {row["name"]: row["value"] for _, row in params_df.iterrows()}


def load_mcmc_params_tables(calib_dirpath: str):
    mcmc_tables = []
    for db_path in _find_db_paths(calib_dirpath):
        db = get_database(db_path)
        mcmc_tables.append(db.query("mcmc_params"))

    return mcmc_tables


def load_mcmc_tables(calib_dirpath: str):
    mcmc_tables = []
    for db_path in _find_db_paths(calib_dirpath):
        db = get_database(db_path)
        mcmc_tables.append(db.query("mcmc_run"))

    return mcmc_tables


def load_uncertainty_table(calib_dirpath: str):
    db_path = _find_db_paths(calib_dirpath)[0]
    db = get_database(db_path)
    return db.query("uncertainty")


def append_tables(tables: List[pd.DataFrame]):
    # TODO: Use this in load_mcmc_tables / load_mcmc_params_tables / load_derived_output_tables
    assert tables
    df = None
    for table_df in tables:
        if df is not None:
            df = df.append(table_df)
        else:
            df = table_df

    return df


def load_derived_output_tables(calib_dirpath: str, column: str = None):
    derived_output_tables = []
    for db_path in _find_db_paths(calib_dirpath):
        db = get_database(db_path)
        if not column:
            df = db.query("derived_outputs")
            derived_output_tables.append(df)
        else:
            cols = ["chain", "run", "scenario", "times", column]
            df = db.query("derived_outputs", columns=cols)
            derived_output_tables.append(df)

    return derived_output_tables


def _find_db_paths(dirpath: str):
    db_paths = []
    for fname in os.listdir(dirpath):
        if fname.startswith("outputs") or fname.startswith("calibration"):
            fpath = os.path.join(dirpath, fname)
            db_paths.append(fpath)

    return sorted(db_paths)
