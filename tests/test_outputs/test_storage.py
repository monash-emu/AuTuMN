from unittest import mock

import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal

from autumn.tb_model.outputs import _unpivot_outputs


def test_unpivot_outputs():
    """
    Verify that store_tb_database 
    """
    mock_model = mock.Mock()
    outputs = [
        [300.0, 300.0, 300.0, 33.0, 33.0, 33.0, 93.0, 39.0],
        [271.0, 300.0, 271.0, 62.0, 33.0, 62.0, 93.0, 69.0],
        [246.0, 300.0, 246.0, 88.0, 33.0, 88.0, 93.0, 89.0],
        [222.0, 300.0, 222.0, 111.0, 33.0, 111.0, 39.0, 119.0],
        [201.0, 300.0, 201.0, 132.0, 33.0, 132.0, 39.0, 139.0],
        [182.0, 300.0, 182.0, 151.0, 33.0, 151.0, 39.0, 159.0],
    ]
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
    mock_model.times = [2000, 2001, 2002, 2003, 2004, 2005]
    mock_model.all_stratifications = {"mood": ["happy", "sad"], "age": ["old", "young"]}
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
    actual_df = _unpivot_outputs(mock_model)
    assert_frame_equal(expected_df, actual_df)
