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


def get_dhhs_vaccination_numbers(
    cluster: str = None, start_age: int = 0, end_age: int = 89, dose=1
):
    """
    Returns number of vaccinations administered in Victoria.
    """
    input_db = get_input_db()

    cond_map = {
        "dosenumber": dose,
        "start_age>": start_age,
        "end_age<": end_age,
    }

    # Conditional to manage state or cluster
    if cluster is None:
        cluster = "Victoria"

    elif cluster is not None:
        cluster = cluster.upper()
        cond_map["cluster_id"] = cluster

    pop = get_pop(cluster, start_age, end_age, input_db)
    df = get_vac_num(input_db, cond_map)

    # Total number of vaccinations per day
    df = df[["date_index", "n"]].groupby(["date_index"], as_index=False).sum()

    # Cumulative vaccination and coverage
    df["cml_n"] = df.n.cumsum()
    df["cml_coverage"] = df.cml_n / pop

    vac_dates = df.date_index.to_numpy()
    coverage_values = df.cml_coverage.to_numpy()
    epsilon = 1e-6  # A really tiny number to avoid having any zeros
    avg_vals = np.array(apply_moving_average(coverage_values, 7)) + epsilon
    return vac_dates, avg_vals


def get_vac_num(input_db, cond_map):
    df = input_db.query(
        "vic_2021", columns=["date_index", "n", "start_age", "end_age"], conditions=cond_map
    )

    return df


def get_pop(cluster, start_age, end_age, input_db):
    year = 2020 if cluster == "Victoria" else 2018
    pop = input_db.query(
        "population",
        columns=["population"],
        conditions={
            "year": year,
            "region": cluster,
            "start_age>": start_age,
            "end_age<": end_age,
        },
    )
    pop = pop.population.sum()
    return pop


def get_yougov_date():
    """ Return the entire YouGov table for Victoria"""
    input_db = get_input_db()
    df = input_db.query("yougov_vic")
    return df