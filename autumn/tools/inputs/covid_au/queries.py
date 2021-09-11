from datetime import date, datetime

import numpy as np
import pandas as pd

from autumn.tools.inputs.database import get_input_db
from autumn.tools.utils.utils import apply_moving_average, COVID_BASE_DATETIME

COVID_BASE_DATE = date(2019, 12, 31)


def get_vic_testing_numbers():
    """
    Returns 7-day moving average of number of tests administered in Victoria.
    """
    input_db = get_input_db()
    df = input_db.query("covid_au", columns=["date", "tests"], conditions={"state_abbrev": "VIC"})
    date_str_to_int = lambda s: (datetime.strptime(s, "%Y-%m-%d") - COVID_BASE_DATETIME).days
    test_dates = df.date.apply(date_str_to_int).to_numpy()
    test_values = df.tests.to_numpy()
    epsilon = 1e-6  # A really tiny number to avoid having any zeros
    avg_vals = np.array(apply_moving_average(test_values, 7)) + epsilon
    return test_dates, avg_vals


def get_dhhs_testing_numbers(cluster: str = None):
    """
    Returns 7-day moving average of number of tests administered in Victoria.
    """
    input_db = get_input_db()

    if cluster is None:
        df = input_db.query("covid_dhhs_test", columns=["date", "test"])
        df = df.groupby("date", as_index=False).sum()
    else:
        df = input_db.query(
            "covid_dhhs_test", columns=["date", "test"], conditions={"cluster_name": cluster}
        )

    test_dates = (pd.to_datetime(df.date) - pd.datetime(2019, 12, 31)).dt.days.to_numpy()
    test_values = df.test.to_numpy()
    epsilon = 1e-6  # A really tiny number to avoid having any zeros
    avg_vals = np.array(apply_moving_average(test_values, 7)) + epsilon
    return test_dates, avg_vals


def get_dhhs_vaccination_numbers(cluster: str = None, agegroup: str = None, dose=1):
    """
    Returns number of vaccinations administered in Victoria.
    """
    input_db = get_input_db()

    if cluster is None and agegroup is None:
        df = input_db.query(
            "vic_2021", columns=["date_index", "n"], conditions={"dosenumber": dose}
        )
        df = df.groupby(["date_index"], as_index=False).sum()
    elif agegroup is None:
        df = input_db.query(
            "vic_2021", columns=["date_index", "n"], conditions={"cluster_id": cluster,"dosenumber": dose}
        )
    date_str_to_int = lambda s: (datetime.strptime(s, "%Y-%m-%d") - COVID_BASE_DATETIME).days

    vac_dates = df.date_index.to_numpy()
    vac_values = df.n.to_numpy()
    epsilon = 1e-6  # A really tiny number to avoid having any zeros
    avg_vals = np.array(apply_moving_average(vac_values, 7)) + epsilon
    return vac_dates, avg_vals
