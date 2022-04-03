"""
Loading data from the output database.
"""
import logging
import os
from typing import List

import numpy as np
import pandas as pd

from summer.model import CompartmentalModel

from ..db.database import BaseDatabase, get_database

logger = logging.getLogger(__name__)


ID_COLS = ["chain", "run", "scenario", "times"]


def load_models_from_database(project, database_path: str) -> List[CompartmentalModel]:
    """
    Attemps to load baseline model and scenario models from a databse file.
    Assumes only 1 run present in database.
    Assumes the model code has not changed.
    """
    out_db = get_database(database_path=database_path)
    models = []
    out_df = out_db.query("outputs")
    do_df = out_db.query("derived_outputs")
    run_names = sorted(out_df["run"].unique().tolist())
    assert len(run_names) == 1, "Expect only 1 run to be present in database."
    scenario_idxs = sorted(out_df["scenario"].unique().tolist())
    scenario_idxs = [int(sc) for sc in scenario_idxs]
    for scenario_idx in scenario_idxs:
        scenario_mask = out_df["scenario"] == scenario_idx
        if scenario_idx == 0:
            # Baseline model
            params = project.param_set.baseline.to_dict()
        else:
            params = project.param_set.scenarios[scenario_idx - 1].to_dict()

        model = project.build_model(params)
        outputs_data = out_df[scenario_mask].to_dict()
        do_data = do_df[scenario_mask].to_dict()
        model.times = np.array(list(outputs_data["times"].values()))
        model.outputs = np.column_stack(
            [list(column.values()) for key, column in outputs_data.items() if key not in ID_COLS]
        )
        model.derived_outputs = {
            key: np.array(list(value.values()))
            for key, value in do_data.items()
            if key not in ID_COLS
        }
        models.append(model)

    return models


def load_mcmc_params(db: BaseDatabase, run_id: int):
    """
    Returns a dict of params
    """
    params_df = db.query("mcmc_params", conditions={"run": run_id})
    return {row["name"]: row["value"] for _, row in params_df.iterrows()}


def load_mcmc_params_tables(calib_dirpath: str):
    mcmc_tables = []
    for db_path in find_db_paths(calib_dirpath):
        db = get_database(db_path)
        mcmc_tables.append(db.query("mcmc_params"))

    return mcmc_tables


def load_mcmc_tables(calib_dirpath: str):
    mcmc_tables = []
    for db_path in find_db_paths(calib_dirpath):
        db = get_database(db_path)
        mcmc_tables.append(db.query("mcmc_run"))

    return mcmc_tables


def load_uncertainty_table(calib_dirpath: str):
    db_path = find_db_paths(calib_dirpath)[0]
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
    for db_path in find_db_paths(calib_dirpath):
        db = get_database(db_path)
        if not column:
            df = db.query("derived_outputs")
            derived_output_tables.append(df)
        else:
            cols = ["chain", "run", "scenario", "times", column]
            df = db.query("derived_outputs", columns=cols)
            derived_output_tables.append(df)

    return derived_output_tables


def find_db_paths(dirpath: str):
    db_paths = []
    for fname in os.listdir(dirpath):
        if fname.startswith("outputs") or fname.startswith("chain") or fname.startswith("powerbi"):
            fpath = os.path.join(dirpath, fname)
            db_paths.append(fpath)

    return sorted(db_paths)
