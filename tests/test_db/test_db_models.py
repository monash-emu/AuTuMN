import os

import pytest
import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal

from ..utils import get_mock_model

from autumn.db import Database
# from autumn.db.models import (
#     unpivot_outputs,
#     load_model_scenarios,
#     collate_outputs,
#     collate_outputs_powerbi,
#     store_run_models,
#     store_database,
#     create_power_bi_outputs,
# )

@pytest.mark.skip("Old code, needs to be re-written")
def test_unpivot_outputs(tmp_path):
    """
    Verify that unpivot_outputs works. 
    """
    out_db_path = os.path.join(tmp_path, "out.db")
    mock_model = get_mock_model(
        times=[2000, 2001, 2002, 2003, 2004, 2005],
        outputs=[
            [300.0, 300.0, 300.0, 33.0, 33.0, 33.0, 93.0, 39.0],
            [271.0, 300.0, 271.0, 62.0, 33.0, 62.0, 93.0, 69.0],
            [246.0, 300.0, 246.0, 88.0, 33.0, 88.0, 93.0, 89.0],
            [222.0, 300.0, 222.0, 111.0, 33.0, 111.0, 39.0, 119.0],
            [201.0, 300.0, 201.0, 132.0, 33.0, 132.0, 39.0, 139.0],
            [182.0, 300.0, 182.0, 151.0, 33.0, 151.0, 39.0, 159.0],
        ],
    )
    store_run_models([mock_model], out_db_path)
    out_db = Database(out_db_path)
    outputs_df = out_db.db_query("outputs")
    unpivoted_df = unpivot_outputs(outputs_df)
    expected_columns = ["idx", "Scenario", "times", "value", "age", "compartment", "mood"]
    expected_data = [
        ["run_0", "S_0", 2000, 300.0, "age_old", "susceptible", "mood_happy"],
        ["run_0", "S_0", 2001, 271.0, "age_old", "susceptible", "mood_happy"],
        ["run_0", "S_0", 2002, 246.0, "age_old", "susceptible", "mood_happy"],
        ["run_0", "S_0", 2003, 222.0, "age_old", "susceptible", "mood_happy"],
        ["run_0", "S_0", 2004, 201.0, "age_old", "susceptible", "mood_happy"],
        ["run_0", "S_0", 2005, 182.0, "age_old", "susceptible", "mood_happy"],
        ["run_0", "S_0", 2000, 300.0, "age_old", "susceptible", "mood_sad"],
        ["run_0", "S_0", 2001, 300.0, "age_old", "susceptible", "mood_sad"],
        ["run_0", "S_0", 2002, 300.0, "age_old", "susceptible", "mood_sad"],
        ["run_0", "S_0", 2003, 300.0, "age_old", "susceptible", "mood_sad"],
        ["run_0", "S_0", 2004, 300.0, "age_old", "susceptible", "mood_sad"],
        ["run_0", "S_0", 2005, 300.0, "age_old", "susceptible", "mood_sad"],
        ["run_0", "S_0", 2000, 300.0, "age_young", "susceptible", "mood_happy"],
        ["run_0", "S_0", 2001, 271.0, "age_young", "susceptible", "mood_happy"],
        ["run_0", "S_0", 2002, 246.0, "age_young", "susceptible", "mood_happy"],
        ["run_0", "S_0", 2003, 222.0, "age_young", "susceptible", "mood_happy"],
        ["run_0", "S_0", 2004, 201.0, "age_young", "susceptible", "mood_happy"],
        ["run_0", "S_0", 2005, 182.0, "age_young", "susceptible", "mood_happy"],
        ["run_0", "S_0", 2000, 33.0, "age_young", "susceptible", "mood_sad"],
        ["run_0", "S_0", 2001, 62.0, "age_young", "susceptible", "mood_sad"],
        ["run_0", "S_0", 2002, 88.0, "age_young", "susceptible", "mood_sad"],
        ["run_0", "S_0", 2003, 111.0, "age_young", "susceptible", "mood_sad"],
        ["run_0", "S_0", 2004, 132.0, "age_young", "susceptible", "mood_sad"],
        ["run_0", "S_0", 2005, 151.0, "age_young", "susceptible", "mood_sad"],
        ["run_0", "S_0", 2000, 33.0, "age_old", "infectious", "mood_happy"],
        ["run_0", "S_0", 2001, 33.0, "age_old", "infectious", "mood_happy"],
        ["run_0", "S_0", 2002, 33.0, "age_old", "infectious", "mood_happy"],
        ["run_0", "S_0", 2003, 33.0, "age_old", "infectious", "mood_happy"],
        ["run_0", "S_0", 2004, 33.0, "age_old", "infectious", "mood_happy"],
        ["run_0", "S_0", 2005, 33.0, "age_old", "infectious", "mood_happy"],
        ["run_0", "S_0", 2000, 33.0, "age_old", "infectious", "mood_sad"],
        ["run_0", "S_0", 2001, 62.0, "age_old", "infectious", "mood_sad"],
        ["run_0", "S_0", 2002, 88.0, "age_old", "infectious", "mood_sad"],
        ["run_0", "S_0", 2003, 111.0, "age_old", "infectious", "mood_sad"],
        ["run_0", "S_0", 2004, 132.0, "age_old", "infectious", "mood_sad"],
        ["run_0", "S_0", 2005, 151.0, "age_old", "infectious", "mood_sad"],
        ["run_0", "S_0", 2000, 93.0, "age_young", "infectious", "mood_happy"],
        ["run_0", "S_0", 2001, 93.0, "age_young", "infectious", "mood_happy"],
        ["run_0", "S_0", 2002, 93.0, "age_young", "infectious", "mood_happy"],
        ["run_0", "S_0", 2003, 39.0, "age_young", "infectious", "mood_happy"],
        ["run_0", "S_0", 2004, 39.0, "age_young", "infectious", "mood_happy"],
        ["run_0", "S_0", 2005, 39.0, "age_young", "infectious", "mood_happy"],
        ["run_0", "S_0", 2000, 39.0, "age_young", "infectious", "mood_sad"],
        ["run_0", "S_0", 2001, 69.0, "age_young", "infectious", "mood_sad"],
        ["run_0", "S_0", 2002, 89.0, "age_young", "infectious", "mood_sad"],
        ["run_0", "S_0", 2003, 119.0, "age_young", "infectious", "mood_sad"],
        ["run_0", "S_0", 2004, 139.0, "age_young", "infectious", "mood_sad"],
        ["run_0", "S_0", 2005, 159.0, "age_young", "infectious", "mood_sad"],
    ]
    expected_df = pd.DataFrame(expected_data, columns=expected_columns)
    assert_frame_equal(expected_df, unpivoted_df)

@pytest.mark.skip("Old code, needs to be re-written")
def test_collate_outputs_powerbi(tmp_path):
    """
    Test the collation of multiple calibration output databases into a single file. 
    """
    # Setup database tables
    mcmc_run_cols = ["idx", "Scenario", "ice_cream_sales", "loglikelihood", "accept"]
    mcmc_run_1 = [
        ["run_0", "S_0", 1, -1, 1],
        ["run_1", "S_0", 2, -2, 1],
        ["run_2", "S_0", 3, -3, 0],
        ["run_3", "S_0", 4, -4, 1],
    ]
    mcmc_run_2 = [
        ["run_0", "S_0", 11, -11, 1],
        ["run_1", "S_0", 12, -12, 0],
        ["run_2", "S_0", 13, -13, 1],
        ["run_3", "S_0", 14, -14, 1],
    ]
    derived_outputs_cols = ["idx", "Scenario", "times", "shark_attacks"]
    derived_outputs_1 = [
        ["run_0", "S_0", 2000, 3],
        ["run_0", "S_0", 2001, 6],
        ["run_0", "S_0", 2002, 10],
        ["run_1", "S_0", 2000, 4],
        ["run_1", "S_0", 2001, 7],
        ["run_1", "S_0", 2002, 11],
        ["run_2", "S_0", 2000, 2],
        ["run_2", "S_0", 2001, 5],
        ["run_2", "S_0", 2002, 9],
        ["run_3", "S_0", 2000, 1],
        ["run_3", "S_0", 2001, 2],
        ["run_3", "S_0", 2002, 3],
    ]
    derived_outputs_2 = [
        ["run_0", "S_0", 2000, 3.1],
        ["run_0", "S_0", 2001, 6.1],
        ["run_0", "S_0", 2002, 10.1],
        ["run_1", "S_0", 2000, 4.1],
        ["run_1", "S_0", 2001, 7.1],
        ["run_1", "S_0", 2002, 11.1],
        ["run_2", "S_0", 2000, 2.1],
        ["run_2", "S_0", 2001, 5.1],
        ["run_2", "S_0", 2002, 9.1],
        ["run_3", "S_0", 2000, 1.1],
        ["run_3", "S_0", 2001, 2.1],
        ["run_3", "S_0", 2002, 3.1],
    ]
    outputs_cols = ["idx", "Scenario", "times", "happy", "sad"]
    outputs_1 = [
        ["run_0", "S_0", 2000, 11, 11],
        ["run_0", "S_0", 2001, 12, 21],
        ["run_0", "S_0", 2002, 13, 31],
        ["run_1", "S_0", 2000, 21, 12],
        ["run_1", "S_0", 2001, 22, 22],
        ["run_1", "S_0", 2002, 23, 32],
        ["run_2", "S_0", 2000, 31, 13],
        ["run_2", "S_0", 2001, 32, 23],
        ["run_2", "S_0", 2002, 33, 33],
        ["run_3", "S_0", 2000, 41, 14],
        ["run_3", "S_0", 2001, 42, 24],
        ["run_3", "S_0", 2002, 43, 34],
    ]
    outputs_2 = [
        ["run_0", "S_0", 2000, 111, 211],
        ["run_0", "S_0", 2001, 112, 221],
        ["run_0", "S_0", 2002, 113, 231],
        ["run_1", "S_0", 2000, 121, 212],
        ["run_1", "S_0", 2001, 122, 222],
        ["run_1", "S_0", 2002, 123, 232],
        ["run_2", "S_0", 2000, 131, 213],
        ["run_2", "S_0", 2001, 132, 223],
        ["run_2", "S_0", 2002, 133, 233],
        ["run_3", "S_0", 2000, 141, 214],
        ["run_3", "S_0", 2001, 142, 224],
        ["run_3", "S_0", 2002, 143, 234],
    ]
    # Create dataframes to save to db
    mcmc_run_1_df = pd.DataFrame(mcmc_run_1, columns=mcmc_run_cols)
    mcmc_run_2_df = pd.DataFrame(mcmc_run_2, columns=mcmc_run_cols)
    derived_ouputs_1_df = pd.DataFrame(derived_outputs_1, columns=derived_outputs_cols)
    derived_ouputs_2_df = pd.DataFrame(derived_outputs_2, columns=derived_outputs_cols)
    outputs_1_df = pd.DataFrame(outputs_1, columns=outputs_cols)
    outputs_2_df = pd.DataFrame(outputs_2, columns=outputs_cols)

    # Connect to test databases
    target_db_path = os.path.join(tmp_path, "target.db")
    db_1_path = os.path.join(tmp_path, f"src-1.db")
    db_2_path = os.path.join(tmp_path, f"src-2.db")
    src_db_paths = [db_1_path, db_2_path]
    target_db = Database(target_db_path)
    src_1_db = Database(db_1_path)
    src_2_db = Database(db_2_path)

    # Save test data to databases
    mcmc_run_1_df.to_sql("mcmc_run", con=src_1_db.engine, index=False)
    mcmc_run_2_df.to_sql("mcmc_run", con=src_2_db.engine, index=False)
    derived_ouputs_1_df.to_sql("derived_outputs", con=src_1_db.engine, index=False)
    derived_ouputs_2_df.to_sql("derived_outputs", con=src_2_db.engine, index=False)
    outputs_1_df.to_sql("outputs", con=src_1_db.engine, index=False)
    outputs_2_df.to_sql("outputs", con=src_2_db.engine, index=False)

    collate_outputs_powerbi(src_db_paths, target_db_path, max_size=0.02)

    expected_mcmc_runs = [
        ["run_0", "S_0", 2, -2, 1],
        ["run_1", "S_0", 4, -4, 1],
        ["run_2", "S_0", 13, -13, 1],
        ["run_3", "S_0", 14, -14, 1],
    ]
    expected_derived_ouputs = [
        ["run_0", "S_0", 2000, 4],
        ["run_0", "S_0", 2001, 7],
        ["run_0", "S_0", 2002, 11],
        ["run_1", "S_0", 2000, 1],
        ["run_1", "S_0", 2001, 2],
        ["run_1", "S_0", 2002, 3],
        ["run_2", "S_0", 2000, 2.1],
        ["run_2", "S_0", 2001, 5.1],
        ["run_2", "S_0", 2002, 9.1],
        ["run_3", "S_0", 2000, 1.1],
        ["run_3", "S_0", 2001, 2.1],
        ["run_3", "S_0", 2002, 3.1],
    ]
    pbi_output_cols = ["idx", "Scenario", "times", "value", "compartment"]
    expected_pbi_outputs = [
        ["run_0", "S_0", 2000, 21, "happy"],
        ["run_0", "S_0", 2001, 22, "happy"],
        ["run_0", "S_0", 2002, 23, "happy"],
        ["run_1", "S_0", 2000, 41, "happy"],
        ["run_1", "S_0", 2001, 42, "happy"],
        ["run_1", "S_0", 2002, 43, "happy"],
        ["run_2", "S_0", 2000, 131, "happy"],
        ["run_2", "S_0", 2001, 132, "happy"],
        ["run_2", "S_0", 2002, 133, "happy"],
        ["run_3", "S_0", 2000, 141, "happy"],
        ["run_3", "S_0", 2001, 142, "happy"],
        ["run_3", "S_0", 2002, 143, "happy"],
        ["run_0", "S_0", 2000, 12, "sad"],
        ["run_0", "S_0", 2001, 22, "sad"],
        ["run_0", "S_0", 2002, 32, "sad"],
        ["run_1", "S_0", 2000, 14, "sad"],
        ["run_1", "S_0", 2001, 24, "sad"],
        ["run_1", "S_0", 2002, 34, "sad"],
        ["run_2", "S_0", 2000, 213, "sad"],
        ["run_2", "S_0", 2001, 223, "sad"],
        ["run_2", "S_0", 2002, 233, "sad"],
        ["run_3", "S_0", 2000, 214, "sad"],
        ["run_3", "S_0", 2001, 224, "sad"],
        ["run_3", "S_0", 2002, 234, "sad"],
    ]
    expected_mcmc_run_df = pd.DataFrame(expected_mcmc_runs, columns=mcmc_run_cols)
    expected_derived_ouputs_df = pd.DataFrame(expected_derived_ouputs, columns=derived_outputs_cols)
    expected_pbi_outputs_df = pd.DataFrame(expected_pbi_outputs, columns=pbi_output_cols)

    # Extract the outputs
    mcmc_df = target_db.db_query("mcmc_run")
    derived_outputs_df = target_db.db_query("derived_outputs")
    pbi_outputs_df = target_db.db_query("pbi_scenario_0")

    # Check that the outputs are correct
    assert_frame_equal(expected_mcmc_run_df, mcmc_df)
    assert_frame_equal(expected_derived_ouputs_df, derived_outputs_df)
    assert_frame_equal(expected_pbi_outputs_df, pbi_outputs_df)

@pytest.mark.skip("Old code, needs to be re-written")
def test_collate_outputs(tmp_path):
    """
    Test the collation of multiple calibration output databases into a single file. 
    """
    # Setup database tables
    mcmc_run_cols = ["idx", "Scenario", "ice_cream_sales", "loglikelihood", "accept"]
    mcmc_run_1 = [
        ["run_0", "S_0", 1, -1, 1],
        ["run_1", "S_0", 2, -2, 1],
        ["run_2", "S_0", 3, -3, 0],
        ["run_3", "S_0", 4, -4, 1],
    ]
    mcmc_run_2 = [
        ["run_0", "S_0", 11, -11, 1],
        ["run_1", "S_0", 12, -12, 0],
        ["run_2", "S_0", 13, -13, 1],
        ["run_3", "S_0", 14, -14, 1],
    ]
    derived_outputs_cols = ["idx", "Scenario", "times", "shark_attacks"]
    derived_outputs_1 = [
        ["run_0", "S_0", 2000, 3],
        ["run_0", "S_0", 2001, 6],
        ["run_0", "S_0", 2002, 10],
        ["run_1", "S_0", 2000, 4],
        ["run_1", "S_0", 2001, 7],
        ["run_1", "S_0", 2002, 11],
        ["run_2", "S_0", 2000, 2],
        ["run_2", "S_0", 2001, 5],
        ["run_2", "S_0", 2002, 9],
        ["run_3", "S_0", 2000, 1],
        ["run_3", "S_0", 2001, 2],
        ["run_3", "S_0", 2002, 3],
    ]
    derived_outputs_2 = [
        ["run_0", "S_0", 2000, 3.1],
        ["run_0", "S_0", 2001, 6.1],
        ["run_0", "S_0", 2002, 10.1],
        ["run_1", "S_0", 2000, 4.1],
        ["run_1", "S_0", 2001, 7.1],
        ["run_1", "S_0", 2002, 11.1],
        ["run_2", "S_0", 2000, 2.1],
        ["run_2", "S_0", 2001, 5.1],
        ["run_2", "S_0", 2002, 9.1],
        ["run_3", "S_0", 2000, 1.1],
        ["run_3", "S_0", 2001, 2.1],
        ["run_3", "S_0", 2002, 3.1],
    ]
    outputs_cols = ["idx", "Scenario", "times", "happy", "sad"]
    outputs_1 = [
        ["run_0", "S_0", 2000, 11, 11],
        ["run_0", "S_0", 2001, 12, 21],
        ["run_0", "S_0", 2002, 13, 31],
        ["run_1", "S_0", 2000, 21, 12],
        ["run_1", "S_0", 2001, 22, 22],
        ["run_1", "S_0", 2002, 23, 32],
        ["run_2", "S_0", 2000, 31, 13],
        ["run_2", "S_0", 2001, 32, 23],
        ["run_2", "S_0", 2002, 33, 33],
        ["run_3", "S_0", 2000, 41, 14],
        ["run_3", "S_0", 2001, 42, 24],
        ["run_3", "S_0", 2002, 43, 34],
    ]
    outputs_2 = [
        ["run_0", "S_0", 2000, 111, 211],
        ["run_0", "S_0", 2001, 112, 221],
        ["run_0", "S_0", 2002, 113, 231],
        ["run_1", "S_0", 2000, 121, 212],
        ["run_1", "S_0", 2001, 122, 222],
        ["run_1", "S_0", 2002, 123, 232],
        ["run_2", "S_0", 2000, 131, 213],
        ["run_2", "S_0", 2001, 132, 223],
        ["run_2", "S_0", 2002, 133, 233],
        ["run_3", "S_0", 2000, 141, 214],
        ["run_3", "S_0", 2001, 142, 224],
        ["run_3", "S_0", 2002, 143, 234],
    ]
    # Create dataframes to save to db
    mcmc_run_1_df = pd.DataFrame(mcmc_run_1, columns=mcmc_run_cols)
    mcmc_run_2_df = pd.DataFrame(mcmc_run_2, columns=mcmc_run_cols)
    derived_ouputs_1_df = pd.DataFrame(derived_outputs_1, columns=derived_outputs_cols)
    derived_ouputs_2_df = pd.DataFrame(derived_outputs_2, columns=derived_outputs_cols)
    outputs_1_df = pd.DataFrame(outputs_1, columns=outputs_cols)
    outputs_2_df = pd.DataFrame(outputs_2, columns=outputs_cols)

    # Connect to test databases
    target_db_path = os.path.join(tmp_path, "target.db")
    db_1_path = os.path.join(tmp_path, f"src-1.db")
    db_2_path = os.path.join(tmp_path, f"src-2.db")
    src_db_paths = [db_1_path, db_2_path]
    target_db = Database(target_db_path)
    src_1_db = Database(db_1_path)
    src_2_db = Database(db_2_path)

    # Save test data to databases
    mcmc_run_1_df.to_sql("mcmc_run", con=src_1_db.engine, index=False)
    mcmc_run_2_df.to_sql("mcmc_run", con=src_2_db.engine, index=False)
    derived_ouputs_1_df.to_sql("derived_outputs", con=src_1_db.engine, index=False)
    derived_ouputs_2_df.to_sql("derived_outputs", con=src_2_db.engine, index=False)
    outputs_1_df.to_sql("outputs", con=src_1_db.engine, index=False)
    outputs_2_df.to_sql("outputs", con=src_2_db.engine, index=False)

    collate_outputs(src_db_paths, target_db_path, num_runs=2)

    expected_mcmc_runs = [
        ["run_0", "S_0", 2, -2, 1],
        ["run_1", "S_0", 4, -4, 1],
        ["run_2", "S_0", 13, -13, 1],
        ["run_3", "S_0", 14, -14, 1],
    ]
    expected_derived_ouputs = [
        ["run_0", "S_0", 2000, 4],
        ["run_0", "S_0", 2001, 7],
        ["run_0", "S_0", 2002, 11],
        ["run_1", "S_0", 2000, 1],
        ["run_1", "S_0", 2001, 2],
        ["run_1", "S_0", 2002, 3],
        ["run_2", "S_0", 2000, 2.1],
        ["run_2", "S_0", 2001, 5.1],
        ["run_2", "S_0", 2002, 9.1],
        ["run_3", "S_0", 2000, 1.1],
        ["run_3", "S_0", 2001, 2.1],
        ["run_3", "S_0", 2002, 3.1],
    ]
    expected_outputs = [
        ["run_0", "S_0", 2000, 21, 12],
        ["run_0", "S_0", 2001, 22, 22],
        ["run_0", "S_0", 2002, 23, 32],
        ["run_1", "S_0", 2000, 41, 14],
        ["run_1", "S_0", 2001, 42, 24],
        ["run_1", "S_0", 2002, 43, 34],
        ["run_2", "S_0", 2000, 131, 213],
        ["run_2", "S_0", 2001, 132, 223],
        ["run_2", "S_0", 2002, 133, 233],
        ["run_3", "S_0", 2000, 141, 214],
        ["run_3", "S_0", 2001, 142, 224],
        ["run_3", "S_0", 2002, 143, 234],
    ]
    expected_mcmc_run_df = pd.DataFrame(expected_mcmc_runs, columns=mcmc_run_cols)
    expected_derived_ouputs_df = pd.DataFrame(expected_derived_ouputs, columns=derived_outputs_cols)
    expected_outputs_df = pd.DataFrame(expected_outputs, columns=outputs_cols)

    # Extract the outputs
    mcmc_df = target_db.db_query("mcmc_run")
    derived_outputs_df = target_db.db_query("derived_outputs")
    outputs_df = target_db.db_query("outputs")

    # Check that the outputs are correct
    assert_frame_equal(expected_mcmc_run_df, mcmc_df)
    assert_frame_equal(expected_derived_ouputs_df, derived_outputs_df)
    assert_frame_equal(expected_outputs_df, outputs_df)

@pytest.mark.skip("Old code, needs to be re-written")
def test_create_power_bi_outputs(tmp_path):
    """
    Ensure that PowerBI outputs are correctly created from a model output database.
    """
    # Prepare models
    models = [
        get_mock_model(
            times=[2000, 2001, 2002, 2003, 2004, 2005],
            outputs=[
                [1, 2, 3, 4, 5, 6, 7, 8],
                [11, 12, 13, 14, 15, 16, 17, 18],
                [21, 22, 23, 24, 25, 26, 27, 28],
                [31, 32, 33, 34, 35, 36, 37, 38],
                [41, 42, 43, 44, 45, 46, 47, 48],
                [5, 4, 3, 2, 1, 0, -1, -2],
            ],
            derived_outputs={
                "times": [2000, 2001, 2002, 2003, 2004, 2005],
                "snacks": [1, 2, 3, 4, 5, 6],
            },
        ),
        get_mock_model(
            times=[2000, 2001, 2002, 2003, 2004, 2005],
            outputs=[
                [51, 52, 53, 54, 55, 56, 57, 58],
                [61, 62, 63, 64, 65, 66, 67, 68],
                [71, 72, 73, 74, 75, 76, 77, 78],
                [81, 82, 83, 94, 95, 96, 97, 98],
                [91, 92, 93, 84, 85, 86, 87, 88],
                [5, 4, 3, 2, 1, 0, -1, -2],
            ],
            derived_outputs={
                "times": [2000, 2001, 2002, 2003, 2004, 2005],
                "snacks": [7, 8, 9, 10, 11, 12],
            },
        ),
    ]
    mcmc_run_df = pd.DataFrame.from_dict(
        {
            "contact_rate": [5, 10, 6, 4],
            "loglikelihood": [-1, -3, -2, -0.5],
            "accept": [1, 0, 0, 1],
        }
    )
    db_path = os.path.join(tmp_path, "out.db")
    powerbi_db_path = os.path.join(tmp_path, "pbi.db")
    # Store the models
    store_run_models(models, db_path)
    store_database(mcmc_run_df, db_path, "mcmc_run", scenario=0, run_idx=1)
    src_db = Database(db_path)
    mcmc_run_src = src_db.db_query("mcmc_run")
    derived_outputs_src = src_db.db_query("derived_outputs")

    # Create Power BI outputs
    create_power_bi_outputs(db_path, powerbi_db_path)
    # Query Power BI outputs
    pbi_db = Database(powerbi_db_path)
    table_0 = pbi_db.db_query("pbi_scenario_0")
    table_1 = pbi_db.db_query("pbi_scenario_1")
    mcmc_run_dest = pbi_db.db_query("mcmc_run")
    derived_outputs_dest = pbi_db.db_query("derived_outputs")

    # Validate derived_outputs copied over
    assert_frame_equal(derived_outputs_src, derived_outputs_dest)

    # Validate MCMC run copied over
    assert_frame_equal(mcmc_run_src, mcmc_run_dest)

    def get_expected_df(model, scenario):
        outputs_df = pd.DataFrame(model.outputs, columns=model.compartment_names)
        outputs_df.insert(0, "times", model.times)
        outputs_df.insert(0, "Scenario", scenario)
        outputs_df.insert(0, "idx", "run_0")
        return unpivot_outputs(outputs_df)

    # Validate Power BI outputs transformed correctly
    expected_df = get_expected_df(models[0], "S_0")
    assert_frame_equal(expected_df, table_0)

    expected_df = get_expected_df(models[1], "S_1")
    assert_frame_equal(expected_df, table_1)

@pytest.mark.skip("Old code, needs to be re-written")
def test_store_and_load_models(tmp_path):
    """
    Ensure that store_run_models actually stores data in a database and that the
    data is in the correct format.
    """
    models = [
        get_mock_model(
            times=[2000, 2001, 2002, 2003, 2004, 2005],
            outputs=[
                [1, 2, 3, 4, 5, 6, 7, 8],
                [11, 12, 13, 14, 15, 16, 17, 18],
                [21, 22, 23, 24, 25, 26, 27, 28],
                [31, 32, 33, 34, 35, 36, 37, 38],
                [41, 42, 43, 44, 45, 46, 47, 48],
                [5, 4, 3, 2, 1, 0, -1, -2],
            ],
            derived_outputs={
                "times": [2000, 2001, 2002, 2003, 2004, 2005],
                "snacks": [1, 2, 3, 4, 5, 6],
            },
        ),
        get_mock_model(
            times=[2000, 2001, 2002, 2003, 2004, 2005],
            outputs=[
                [51, 52, 53, 54, 55, 56, 57, 58],
                [61, 62, 63, 64, 65, 66, 67, 68],
                [71, 72, 73, 74, 75, 76, 77, 78],
                [81, 82, 83, 94, 95, 96, 97, 98],
                [91, 92, 93, 84, 85, 86, 87, 88],
                [5, 4, 3, 2, 1, 0, -1, -2],
            ],
            derived_outputs={
                "times": [2000, 2001, 2002, 2003, 2004, 2005],
                "snacks": [7, 8, 9, 10, 11, 12],
            },
        ),
    ]
    db_path = os.path.join(tmp_path, "out.db")
    # Store the models
    store_run_models(models, db_path)
    # Retrieve the models
    scenarios = load_model_scenarios(db_path)

    for idx, original_model in enumerate(models):
        scenario_model = scenarios[idx].model

        # Check times are the same
        assert (scenario_model.times == np.array(original_model.times)).all()

        # Check loaded outputs are the same as stored outputs
        assert (scenario_model.outputs == np.array(original_model.outputs)).all()

        # Check derived outputs are the same as stored outputs
        assert scenario_model.derived_outputs["snacks"] == original_model.derived_outputs["snacks"]
