import os
import random
from unittest import mock

from summer import Compartment, Stratification

import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def get_mock_model(times, outputs, derived_outputs=None):
    mock_model = mock.Mock()
    mock_model.derived_outputs = derived_outputs or {}
    mock_model.outputs = np.array(outputs)
    mock_model.compartments = [
        Compartment("susceptible", {"mood": "happy", "age": "old"}),
        Compartment("susceptible", {"mood": "sad", "age": "old"}),
        Compartment("susceptible", {"mood": "happy", "age": "young"}),
        Compartment("susceptible", {"mood": "sad", "age": "young"}),
        Compartment("infectious", {"mood": "happy", "age": "old"}),
        Compartment("infectious", {"mood": "sad", "age": "old"}),
        Compartment("infectious", {"mood": "happy", "age": "young"}),
        Compartment("infectious", {"mood": "sad", "age": "young"}),
    ]
    mock_model.times = times
    mock_model._stratifications = [
        Stratification("mood", ["happy", "sad"], ["susceptible", "infectious"]),
        Stratification("age", ["old", "young"], ["susceptible", "infectious"]),
    ]
    return mock_model


def get_deterministic_random_seed(num: int):
    return num


def in_memory_db_factory():
    """Replacement for get_sql_engine, returns an in-memory database"""
    databases = {}

    def get_in_memory_db_engine(db_path: str):
        """
        Returns an in-memory SQLite database that corresponds to db_path.
        """
        assert db_path.endswith(".db"), f'Database path "{db_path}" must be a file that ends in .db'

        if db_path.endswith("inputs.db"):
            # Return the real "inputs.db" if it's requested
            rel_db_path = os.path.relpath(db_path)
            engine = create_engine(f"sqlite:///{rel_db_path}", echo=False)
            return engine

        # Return an in-memory SQL Alchemy SQLite database engine.
        try:
            return databases[db_path]
        except KeyError:
            # Create a fake DB path.
            if not os.path.exists(db_path):
                with open(db_path, "w"):
                    pass

            # Create an in-memory SQLite engine
            engine = create_engine("sqlite://", echo=False)
            databases[db_path] = engine
            return engine

    return get_in_memory_db_engine


def build_synthetic_calibration(targets: dict, funcs: list, chains: int, runs: int, times: int):
    chains = list(range(chains))  # Simulate calibration chains
    runs = list(range(runs))  # Runs per chain
    times = list(range(times))  # Timesteps per run
    outputs = [o["output_key"] for o in targets.values()]

    # Build dataframes for database tables.
    do_columns = ["chain", "run", "scenario", "times"]
    do_data = {"chain": [], "run": [], "scenario": [], "times": []}
    for o in outputs:
        do_columns.append(o)
        do_data[o] = []

    mcmc_columns = ["chain", "run", "loglikelihood", "ap_loglikelihood", "accept", "weight"]
    mcmc_data = {
        "chain": [],
        "run": [],
        "loglikelihood": [],
        "ap_loglikelihood": [],
        "accept": [],
        "weight": [],
    }

    params = ["bar", "baz"]
    params_columns = ["chain", "run", "name", "value"]
    params_data = {
        "chain": [],
        "run": [],
        "name": [],
        "value": [],
    }

    # Create synthetic data
    for chain in chains:
        last_accept_idx = 0
        for run_idx, run in enumerate(runs):
            # Simulate filling the mcmc_run table.
            mcmc_data["chain"].append(chain)
            mcmc_data["run"].append(run)
            mcmc_data["loglikelihood"].append(-1 * random.random())
            mcmc_data["ap_loglikelihood"].append(-1 * random.random())

            for param in params:
                params_data["chain"].append(chain)
                params_data["run"].append(run)
                params_data["name"].append(param)
                params_data["value"].append(random.random())

            is_accepted = random.random() > 0.6
            if not is_accepted:
                accept = 0
                weight = 0
            else:
                accept = 1
                weight = 1
                idx = run_idx - last_accept_idx
                last_accept_idx = run_idx
                if mcmc_data["weight"]:
                    mcmc_data["weight"][-idx] = idx

            mcmc_data["weight"].append(weight)
            mcmc_data["accept"].append(accept)
            for time in times:
                # Simulate filling the derived_outputs table.
                do_data["chain"].append(chain)
                do_data["run"].append(run)
                do_data["scenario"].append(0)
                do_data["times"].append(time)
                for idx, o in enumerate(outputs):
                    do_data[o].append(funcs[idx](time))

    do_df = pd.DataFrame(columns=do_columns, data=do_data)
    mcmc_df = pd.DataFrame(columns=mcmc_columns, data=mcmc_data)
    params_df = pd.DataFrame(columns=params_columns, data=params_data)
    return do_df, mcmc_df, params_df
