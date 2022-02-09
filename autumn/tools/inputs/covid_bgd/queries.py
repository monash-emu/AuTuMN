from datetime import date, datetime

import numpy as np
import pandas as pd

from autumn.tools.inputs.database import get_input_db

TINY_NUMBER = 1e-6


def get_coxs_bazar_testing_numbers():
    """
    Returns 7-day moving average of number of tests administered in Cox's bazar.
    """
    input_db = get_input_db()
    df = input_db.query("covid_coxs_bazar", columns=["date_index", "test"])
    df.dropna(inplace=True)
    test_dates = df.date_index.to_numpy()
    avg_vals = df["test"].to_numpy() + TINY_NUMBER

    return test_dates, avg_vals
