"""
Storing model data in the database
"""
import logging
from typing import List

import yaml
import pandas as pd

from summer.legacy.model import StratifiedModel
from summer.model import CompartmentalModel
from autumn.db.database import get_database

from . import process

logger = logging.getLogger(__name__)


class Table:
    MCMC = "mcmc_run"
    PARAMS = "mcmc_params"
    OUTPUTS = "outputs"
    DERIVED = "derived_outputs"


def save_mle_params(database_path: str, target_path: str):
    """
    Saves the MCMC parameters for the MLE run as a YAML file in the target path.
    """
    db = get_database(database_path)
    mcmc_df = db.query("mcmc_run")
    param_df = db.query("mcmc_params")
    mle_params = process.find_mle_params(mcmc_df, param_df)
    with open(target_path, "w") as f:
        yaml.dump(mle_params, f)


def build_outputs_table(models: List[StratifiedModel], run_id: int, chain_id=None):
    outputs_df = None
    for idx, model in enumerate(models):
        if type(model) is CompartmentalModel:
            names = [str(c) for c in model.compartments]
        else:
            names = model.compartment_names

        # Save model outputs
        df = pd.DataFrame(model.outputs, columns=names)
        df.insert(0, column="chain", value=chain_id)
        df.insert(1, column="run", value=run_id)
        df.insert(2, column="scenario", value=idx)
        df.insert(3, column="times", value=model.times)
        if outputs_df is not None:
            outputs_df = outputs_df.append(df, ignore_index=True)
        else:
            outputs_df = df

    return outputs_df


def build_derived_outputs_table(models: List[StratifiedModel], run_id: int, chain_id=None):
    derived_outputs_df = None
    for idx, model in enumerate(models):
        # Save model derived outputs
        df = pd.DataFrame.from_dict(model.derived_outputs)
        df.insert(0, column="chain", value=chain_id)
        df.insert(1, column="run", value=run_id)
        df.insert(2, column="scenario", value=idx)
        df.insert(3, column="times", value=model.times)
        if derived_outputs_df is not None:
            derived_outputs_df = derived_outputs_df.append(df, ignore_index=True)
        else:
            derived_outputs_df = df

    return derived_outputs_df
