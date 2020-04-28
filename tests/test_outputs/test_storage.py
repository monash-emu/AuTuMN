import os
from unittest import mock
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal

from autumn.tb_model.outputs import (
    unpivot_outputs,
    load_model_scenarios,
    store_run_models,
    create_power_bi_outputs,
    Database,
)


def test_unpivot_outputs():
    """
    Verify that unpivot_outputs works. 
    """
    mock_model = _get_mock_model(
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
    expected_columns = ["times", "value", "compartment", "mood", "age"]
    expected_data = [
        [2000, 300.0, "susceptible", "mood_happy", "age_old"],
        [2001, 271.0, "susceptible", "mood_happy", "age_old"],
        [2002, 246.0, "susceptible", "mood_happy", "age_old"],
        [2003, 222.0, "susceptible", "mood_happy", "age_old"],
        [2004, 201.0, "susceptible", "mood_happy", "age_old"],
        [2005, 182.0, "susceptible", "mood_happy", "age_old"],
        [2000, 300.0, "susceptible", "mood_sad", "age_old"],
        [2001, 300.0, "susceptible", "mood_sad", "age_old"],
        [2002, 300.0, "susceptible", "mood_sad", "age_old"],
        [2003, 300.0, "susceptible", "mood_sad", "age_old"],
        [2004, 300.0, "susceptible", "mood_sad", "age_old"],
        [2005, 300.0, "susceptible", "mood_sad", "age_old"],
        [2000, 300.0, "susceptible", "mood_happy", "age_young"],
        [2001, 271.0, "susceptible", "mood_happy", "age_young"],
        [2002, 246.0, "susceptible", "mood_happy", "age_young"],
        [2003, 222.0, "susceptible", "mood_happy", "age_young"],
        [2004, 201.0, "susceptible", "mood_happy", "age_young"],
        [2005, 182.0, "susceptible", "mood_happy", "age_young"],
        [2000, 33.0, "susceptible", "mood_sad", "age_young"],
        [2001, 62.0, "susceptible", "mood_sad", "age_young"],
        [2002, 88.0, "susceptible", "mood_sad", "age_young"],
        [2003, 111.0, "susceptible", "mood_sad", "age_young"],
        [2004, 132.0, "susceptible", "mood_sad", "age_young"],
        [2005, 151.0, "susceptible", "mood_sad", "age_young"],
        [2000, 33.0, "infectious", "mood_happy", "age_old"],
        [2001, 33.0, "infectious", "mood_happy", "age_old"],
        [2002, 33.0, "infectious", "mood_happy", "age_old"],
        [2003, 33.0, "infectious", "mood_happy", "age_old"],
        [2004, 33.0, "infectious", "mood_happy", "age_old"],
        [2005, 33.0, "infectious", "mood_happy", "age_old"],
        [2000, 33.0, "infectious", "mood_sad", "age_old"],
        [2001, 62.0, "infectious", "mood_sad", "age_old"],
        [2002, 88.0, "infectious", "mood_sad", "age_old"],
        [2003, 111.0, "infectious", "mood_sad", "age_old"],
        [2004, 132.0, "infectious", "mood_sad", "age_old"],
        [2005, 151.0, "infectious", "mood_sad", "age_old"],
        [2000, 93.0, "infectious", "mood_happy", "age_young"],
        [2001, 93.0, "infectious", "mood_happy", "age_young"],
        [2002, 93.0, "infectious", "mood_happy", "age_young"],
        [2003, 39.0, "infectious", "mood_happy", "age_young"],
        [2004, 39.0, "infectious", "mood_happy", "age_young"],
        [2005, 39.0, "infectious", "mood_happy", "age_young"],
        [2000, 39.0, "infectious", "mood_sad", "age_young"],
        [2001, 69.0, "infectious", "mood_sad", "age_young"],
        [2002, 89.0, "infectious", "mood_sad", "age_young"],
        [2003, 119.0, "infectious", "mood_sad", "age_young"],
        [2004, 139.0, "infectious", "mood_sad", "age_young"],
        [2005, 159.0, "infectious", "mood_sad", "age_young"],
    ]
    expected_df = pd.DataFrame(expected_data, columns=expected_columns)
    actual_df = unpivot_outputs(mock_model)
    assert_frame_equal(expected_df, actual_df)


def test_create_power_bi_outputs():
    """
    Ensure that PowerBI outputs are correctly created from a model output database.
    """
    # Prepare models
    models = [
        _get_mock_model(
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
        _get_mock_model(
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
    with TemporaryDirectory() as dirpath:
        db_path = os.path.join(dirpath, "out.db")
        powerbi_db_path = os.path.join(dirpath, "pbi.db")
        # Store the models
        store_run_models(models, db_path)
        # Create Power BI outputs
        create_power_bi_outputs(db_path, powerbi_db_path)
        # Query Power BI outputs
        pbi_db = Database(powerbi_db_path)
        table_0 = pbi_db.db_query("pbi_scenario_0")
        table_1 = pbi_db.db_query("pbi_scenario_1")

    # Validate Power BI outputs
    expected_df = unpivot_outputs(models[0])
    expected_df.insert(0, "Scenario", "S_0")
    expected_df.insert(0, "idx", "run_0")
    assert_frame_equal(expected_df, table_0)

    expected_df = unpivot_outputs(models[1])
    expected_df.insert(0, "Scenario", "S_1")
    expected_df.insert(0, "idx", "run_0")
    assert_frame_equal(expected_df, table_1)


def test_store_and_load_models():
    """
    Ensure that store_run_models actually stores data in a database and that the
    data is in the correct format.
    """
    models = [
        _get_mock_model(
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
        _get_mock_model(
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
    with TemporaryDirectory() as dirpath:
        db_path = os.path.join(dirpath, "out.db")
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


def _get_mock_model(times, outputs, derived_outputs=None):
    mock_model = mock.Mock()
    mock_model.derived_outputs = derived_outputs or {}
    mock_model.outputs = np.array(outputs)
    mock_model.compartment_names = [
        "susceptibleXmood_happyXage_old",
        "susceptibleXmood_sadXage_old",
        "susceptibleXmood_happyXage_young",
        "susceptibleXmood_sadXage_young",
        "infectiousXmood_happyXage_old",
        "infectiousXmood_sadXage_old",
        "infectiousXmood_happyXage_young",
        "infectiousXmood_sadXage_young",
    ]
    mock_model.times = times
    mock_model.all_stratifications = {"mood": ["happy", "sad"], "age": ["old", "young"]}
    return mock_model
