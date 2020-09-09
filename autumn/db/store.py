"""
Storing model data in the database
"""
import logging
from typing import List

import pandas as pd

from summer.model import StratifiedModel
from autumn.db.database import Database


logger = logging.getLogger(__name__)


def increment_mcmc_weight(
    database_path: str, run_id: int, chain_id: int,
):
    logger.info("Incrementing %s %s", run_id, chain_id)
    db = Database(database_path)
    sql = f"UPDATE mcmc_run SET weight = weight + 1 WHERE chain={chain_id} AND run={run_id}"
    db.engine.execute(sql)


def store_mcmc_run(
    database_path: str,
    run_id: int,
    chain_id: int,
    weight: int,
    loglikelihood: float,
    ap_loglikelihood: float,
    accept: int,
    params: dict,
):
    db = Database(database_path)
    # Write run progress.
    columns = ["chain", "run", "loglikelihood", "ap_loglikelihood", "accept", "weight"]
    data = {
        "chain": [chain_id],
        "run": [run_id],
        "loglikelihood": [loglikelihood],
        "ap_loglikelihood": [ap_loglikelihood],
        "accept": [accept],
        "weight": [weight],
    }
    df = pd.DataFrame(data=data, columns=columns)
    db.dump_df("mcmc_run", df)

    if accept:
        # Write run parameters.
        columns = ["chain", "run", "name", "value"]
        run_ids, chain_ids, names, values = [], [], [], []
        for k, v in params.items():
            run_ids.append(run_id)
            chain_ids.append(chain_id)
            names.append(k)
            values.append(v)

        data = {
            "chain": chain_ids,
            "run": run_ids,
            "name": names,
            "value": values,
        }
        df = pd.DataFrame(data=data, columns=columns)
        db.dump_df("mcmc_params", df)


def store_run_models(models: List[StratifiedModel], database_path: str, run_id: int, chain_id=None):
    """
    Store models in the database.
    Assume that models are sorted in an order such that their index is their scenario idx.
    """
    db = Database(database_path)
    for idx, model in enumerate(models):
        # Save model outputs
        df = pd.DataFrame(model.outputs, columns=model.compartment_names)
        df.insert(0, column="chain", value=chain_id)
        df.insert(1, column="run", value=run_id)
        df.insert(2, column="scenario", value=idx)
        df.insert(3, column="times", value=model.times)
        db.dump_df("outputs", df)

        # Save model derived outputs
        df = pd.DataFrame.from_dict(model.derived_outputs)
        df.insert(0, column="chain", value=chain_id)
        df.insert(1, column="run", value=run_id)
        df.insert(2, column="scenario", value=idx)
        df.insert(3, column="times", value=model.times)
        db.dump_df("derived_outputs", df)
