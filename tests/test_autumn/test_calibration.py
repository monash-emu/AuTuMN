import os
from unittest.mock import patch

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_almost_equal

from autumn.tools import db
from autumn.runners.calibration import Calibration, CalibrationMode
from autumn.calibration.utils import (
    sample_starting_params_from_lhs,
    specify_missing_prior_params,
)
from tests.utils import get_mock_model


def test_sample_starting_params_from_lhs__with_lognormal_prior_and_one_sample():
    priors = [
        {
            "param_name": "ice_cream_sales",
            "distribution": "lognormal",
            "distri_params": [-1, 1],
        }
    ]
    specify_missing_prior_params(priors)
    params = sample_starting_params_from_lhs(priors, n_samples=1)
    assert_almost_equal(params[0]['ice_cream_sales'],  0.36787944117144233)


def test_sample_starting_params_from_lhs__with_beta_prior_and_one_sample():
    priors = [
        {
            "param_name": "ice_cream_sales",
            "distribution": "beta",
            "distri_mean": 0.05,
            "distri_ci": [0.01, 0.1],
        }
    ]
    specify_missing_prior_params(priors)
    params = sample_starting_params_from_lhs(priors, n_samples=1)
    assert_almost_equal(params[0]['ice_cream_sales'],  0.04680260472064115)


def test_sample_starting_params_from_lhs__with_gamma_prior_and_one_sample():
    priors = [
        {
            "param_name": "ice_cream_sales",
            "distribution": "gamma",
            "distri_mean": 5.0,
            "distri_ci": [3.0, 7.0],
        }
    ]
    specify_missing_prior_params(priors)
    params = sample_starting_params_from_lhs(priors, n_samples=1)
    assert_almost_equal(params[0]['ice_cream_sales'],  4.932833078981056)


def test_sample_starting_params_from_lhs__with_uniform_prior_and_one_sample():
    priors = [{"param_name": "ice_cream_sales", "distribution": "uniform", "distri_params": [1, 5]}]
    specify_missing_prior_params(priors)
    params = sample_starting_params_from_lhs(priors, n_samples=1)
    assert _prepare_params(params) == _prepare_params([{"ice_cream_sales": 3.0}])


def test_sample_starting_params_from_lhs__with_uniform_priors_and_one_sample():
    priors = [
        {"param_name": "ice_cream_sales", "distribution": "uniform", "distri_params": [1, 5]},
        {"param_name": "air_temp", "distribution": "uniform", "distri_params": [1, 10]},
    ]
    specify_missing_prior_params(priors)
    params = sample_starting_params_from_lhs(priors, n_samples=1)
    assert _prepare_params(params) == _prepare_params([{"ice_cream_sales": 3.0, "air_temp": 5.5}])


def test_sample_starting_params_from_lhs__with_uniform_prior_and_two_samples():
    priors = [{"param_name": "ice_cream_sales", "distribution": "uniform", "distri_params": [1, 5]}]
    specify_missing_prior_params(priors)
    params = sample_starting_params_from_lhs(priors, n_samples=2)
    assert _prepare_params(params) == _prepare_params(
        [{"ice_cream_sales": 2.0}, {"ice_cream_sales": 4.0}]
    )


def test_sample_starting_params_from_lhs__with_uniform_priors_and_two_samples():
    priors = [
        {"param_name": "ice_cream_sales", "distribution": "uniform", "distri_params": [1, 5]},
        {"param_name": "air_temp", "distribution": "uniform", "distri_params": [1, 10]},
    ]
    specify_missing_prior_params(priors)
    params = sample_starting_params_from_lhs(priors, n_samples=2)
    assert _prepare_params(params) == _prepare_params(
        [{"ice_cream_sales": 2.0, "air_temp": 3.25}, {"ice_cream_sales": 4.0, "air_temp": 7.75}],
    ) or _prepare_params(params) == _prepare_params(
        [{"ice_cream_sales": 4.0, "air_temp": 3.25}, {"ice_cream_sales": 2.0, "air_temp": 7.75}],
    )


def test_sample_starting_params_from_lhs__with_uniform_prior_and_four_samples():
    priors = [{"param_name": "ice_cream_sales", "distribution": "uniform", "distri_params": [1, 5]}]
    specify_missing_prior_params(priors)
    params = sample_starting_params_from_lhs(priors, n_samples=4)
    assert _prepare_params(params) == _prepare_params(
        [
            {"ice_cream_sales": 1.5},
            {"ice_cream_sales": 2.5},
            {"ice_cream_sales": 3.5},
            {"ice_cream_sales": 4.5},
        ],
    )


def _prepare_params(l):
    return set([tuple(sorted(ps.items())) for ps in l])


from autumn.calibration.priors import UniformPrior
from autumn.calibration.targets import PoissonTarget
from autumn.tools.project import Project, ParameterSet, Params


def test_calibrate_autumn_mcmc(temp_data_dir):
    priors = [UniformPrior("ice_cream_sales", [1, 5])]
    target_outputs = [
        PoissonTarget(
            pd.Series(
                [3, 6, 9, 12, 15], [2000, 2001, 2002, 2003, 2004], name="shark_attacks"
            )
        ),
    ]
    calib = Calibration(priors, target_outputs, seed=0)
    calib._no_pickle = True
    params = ParameterSet(Params({"ice_cream_sales": 0, "time": {"start": 2000}}))
    project = Project("hawaii", "sharks", _build_mock_model, params, calib)
    calib.run(project, 1, 1, 1)

    app_dir = os.path.join(temp_data_dir, "outputs", "calibrate", "sharks", "hawaii")
    run_dir = os.path.join(app_dir, os.listdir(app_dir)[0])
    db_paths = db.load.find_db_paths(run_dir)
    assert len(db_paths) == 1
    out_db = db.get_database(db_paths[0])
    assert set(out_db.table_names()) == {
        "derived_outputs",
        "mcmc_run",
        "mcmc_params",
    }
    mcmc_runs = out_db.query("mcmc_run")
    mcmc_params = out_db.query("mcmc_params")
    mle_params = db.process.find_mle_params(mcmc_runs, mcmc_params)
    ice_cream_sales_mle = mle_params["ice_cream_sales"]
    # +++FIXME Should be deterministic now
    assert 2.9 < ice_cream_sales_mle < 3.1


def _build_mock_model(params, build_options=None):
    """
    Fake model building function where derived output "shark_attacks"
    is influenced by the ice_cream_sales input parameter.
    """
    ice_cream_sales = params["ice_cream_sales"]
    vals = [0, 1, 2, 3, 4, 5]
    mock_model = get_mock_model(
        times=np.array([1999, 2000, 2001, 2002, 2003, 2004]),
        outputs=[
            [300.0, 300.0, 300.0, 33.0, 33.0, 33.0, 93.0, 39.0],
            [271.0, 300.0, 271.0, 62.0, 33.0, 62.0, 93.0, 69.0],
            [246.0, 300.0, 246.0, 88.0, 33.0, 88.0, 93.0, 89.0],
            [222.0, 300.0, 222.0, 111.0, 33.0, 111.0, 39.0, 119.0],
            [201.0, 300.0, 201.0, 132.0, 33.0, 132.0, 39.0, 139.0],
            [182.0, 300.0, 182.0, 151.0, 33.0, 151.0, 39.0, 159.0],
        ],
        derived_outputs={
            "shark_attacks": np.array([ice_cream_sales * i for i in vals]),
        },
    )
    return mock_model
