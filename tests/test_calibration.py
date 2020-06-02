import os
from copy import deepcopy

import pytest
from autumn.db import Database
from autumn.calibration import Calibration, CalibrationMode

from .utils import get_mock_model


@pytest.mark.skip("Old code, needs to be re-written")
@pytest.mark.parametrize("num_iters, max_size_mb", [(50, 0.1), (50, 0.2), (50, 0.3)])
def test_calibration_data_collation(temp_data_dir, num_iters, max_size_mb):
    """
    Test that collate_powerbi_database correctly pulls data from calibration outputs into a single database.
    """
    priors = [
        {
            "param_name": "ice_cream_sales",
            "distribution": "uniform",
            "distri_params": [1, 5],
        }
    ]
    target_outputs = [
        {
            "output_key": "shark_attacks",
            "years": [2000, 2001, 2002, 2003, 2004],
            "values": [3, 6, 9, 12, 15],
            "loglikelihood_distri": "poisson",
        }
    ]

    for chain_index in range(5):
        calib = Calibration(
            "sharks",
            _build_mock_model,
            deepcopy(priors),
            deepcopy(target_outputs),
            multipliers={},
            chain_index=chain_index,
            model_parameters={
                "default": {},
                "scenario_start_time": 2000,
                "scenarios": {},
            },
        )
        calib.run_fitting_algorithm(
            run_mode=CalibrationMode.AUTUMN_MCMC,
            n_iterations=num_iters,
            n_burned=10,
            n_chains=1,
            available_time=1e6,
        )

    app_dir = os.path.join(temp_data_dir, "sharks")
    run_dir = os.path.join(app_dir, os.listdir(app_dir)[0])
    out_db_paths = [
        os.path.join(run_dir, db_path)
        for db_path in os.listdir(run_dir)
        if db_path.endswith(".db")
    ]
    # There should be 5 databases saved
    assert len(out_db_paths) == 5

    # Collate the outputs into one database
    target_db_path = os.path.join(temp_data_dir, "collated.db")
    collate_outputs_powerbi(out_db_paths, target_db_path, max_size=max_size_mb)

    target_db = Database(target_db_path)
    collated_mcmc_runs = target_db.db_query("mcmc_run")
    assert len(collated_mcmc_runs) == collated_mcmc_runs.accept.sum()
    assert max_size_mb / 2 <= target_db.get_size_mb() <= max_size_mb


def test_calibrate_autumn_mcmc(temp_data_dir):
    # Import autumn stuff inside function so we can mock out the database.
    priors = [
        {
            "param_name": "ice_cream_sales",
            "distribution": "uniform",
            "distri_params": [1, 5],
        }
    ]
    target_outputs = [
        {
            "output_key": "shark_attacks",
            "years": [2000, 2001, 2002, 2003, 2004],
            "values": [3, 6, 9, 12, 15],
            "loglikelihood_distri": "poisson",
        }
    ]
    calib = Calibration(
        "sharks",
        _build_mock_model,
        priors,
        target_outputs,
        multipliers={},
        chain_index=0,
        model_parameters={"default": {}, "scenario_start_time": 2000, "scenarios": {}},
    )
    calib.run_fitting_algorithm(
        run_mode=CalibrationMode.AUTUMN_MCMC,
        n_iterations=50,
        n_burned=10,
        n_chains=1,
        available_time=1e6,
    )
    app_dir = os.path.join(temp_data_dir, "sharks")

    run_dir = os.path.join(app_dir, os.listdir(app_dir)[0])
    db_fname = [fname for fname in os.listdir(run_dir) if fname.endswith(".db")][0]
    out_db_path = os.path.join(run_dir, db_fname)
    assert os.path.exists(out_db_path)

    out_db = Database(out_db_path)
    assert set(out_db.engine.table_names()) == {
        "outputs",
        "derived_outputs",
        "mcmc_run",
    }
    mcmc_runs = out_db.db_query("mcmc_run")
    max_idx = mcmc_runs.loglikelihood.idxmax()
    best_run = mcmc_runs.iloc[max_idx]
    ice_cream_sales_mle = best_run.ice_cream_sales
    # This value is non-deterministic due to fixed seed.
    assert 2.9 < ice_cream_sales_mle < 3.1


def _build_mock_model(params):
    """
    Fake model building function where derived output "shark_attacks" 
    is influenced by the ice_cream_sales input parameter.
    """
    ice_cream_sales = params["ice_cream_sales"]
    vals = [0, 1, 2, 3, 4, 5]
    mock_model = get_mock_model(
        times=[1999, 2000, 2001, 2002, 2003, 2004],
        outputs=[
            [300.0, 300.0, 300.0, 33.0, 33.0, 33.0, 93.0, 39.0],
            [271.0, 300.0, 271.0, 62.0, 33.0, 62.0, 93.0, 69.0],
            [246.0, 300.0, 246.0, 88.0, 33.0, 88.0, 93.0, 89.0],
            [222.0, 300.0, 222.0, 111.0, 33.0, 111.0, 39.0, 119.0],
            [201.0, 300.0, 201.0, 132.0, 33.0, 132.0, 39.0, 139.0],
            [182.0, 300.0, 182.0, 151.0, 33.0, 151.0, 39.0, 159.0],
        ],
        derived_outputs={
            "times": [1999, 2000, 2001, 2002, 2003, 2004],
            "shark_attacks": [ice_cream_sales * i for i in vals],
        },
    )
    return mock_model
