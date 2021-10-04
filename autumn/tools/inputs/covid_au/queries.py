from datetime import date, datetime

import numpy as np
import pandas as pd

from autumn.tools.inputs.database import get_input_db
from autumn.tools.utils.utils import apply_moving_average, COVID_BASE_DATETIME

COVID_BASE_DATE = date(2019, 12, 31)
TINY_NUMBER = 1e-6


def get_vic_testing_numbers():
    """
    Returns 7-day moving average of number of tests administered in Victoria.
    """
    input_db = get_input_db()
    df = input_db.query("covid_au", columns=["date", "tests"], conditions={"state_abbrev": "VIC"})
    date_str_to_int = lambda s: (datetime.strptime(s, "%Y-%m-%d") - COVID_BASE_DATETIME).days
    test_dates = df.date.apply(date_str_to_int).to_numpy()
    test_values = df.tests.to_numpy()
    avg_vals = np.array(apply_moving_average(test_values, 7)) + TINY_NUMBER
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
    avg_vals = np.array(apply_moving_average(test_values, 7)) + TINY_NUMBER
    return test_dates, avg_vals


def get_historical_vac_coverage(
    cluster: str = None, start_age: int = 0, end_age: int = 89, dose=1
):
    """
    Returns vaccination coverage in Victorian health cluster.
    """
    input_db = get_input_db()

    cond_map = {
        "dosenumber": dose,
        "start_age>": start_age,
        "end_age<": end_age,
    }

    # Conditional to manage state or cluster
    cond_map = update_cond_map(cluster, cond_map)

    pop = get_pop(cluster, start_age, end_age, input_db)
    df = get_historical_vac_num(input_db, cond_map)

    # Total number of vaccinations per day
    df = df[["date_index", "n"]].groupby(["date_index"], as_index=False).sum()

    # Cumulative vaccination and coverage
    df["cml_n"] = df.n.cumsum()
    df["cml_coverage"] = df.cml_n / pop

    vac_dates = df.date_index.to_numpy()
    coverage_values = df.cml_coverage.to_numpy()
    avg_vals = np.array(apply_moving_average(coverage_values, 7)) + TINY_NUMBER
    return vac_dates, avg_vals

def update_cond_map(cluster, cond_map):
    if cluster is None:
        cluster = "Victoria"

    elif cluster is not None:
        cluster = cluster.upper()
        cond_map["cluster_id"] = cluster
    return cond_map


def get_historical_vac_num(input_db, cond_map):
    df = input_db.query(
        "vic_2021", columns=["date_index", "n", "start_age", "end_age"], conditions=cond_map
    )

    return df


def get_modelled_vac_num(input_db, cond_map, dose):
    df = input_db.query(
        "vac_model", columns=["date_index", dose, "start_age", "end_age"], conditions=cond_map
    )

    return df


def get_modelled_vac_coverage(
    cluster: str = None, start_age: int = 0, end_age: int = 89, vaccine="pfizer", dose='dose_1'
):
    """
    Returns number of vaccinations administered in Victoria.
    """
    input_db = get_input_db()

    cond_map = {
        "vaccine_brand_name": vaccine,
        "start_age>": start_age,
        "end_age<": end_age,
    }

    # Conditional to manage state or cluster
    cond_map = update_cond_map(cluster, cond_map)

    pop = get_pop(cluster, start_age, end_age, input_db)
    df = get_modelled_vac_num(input_db, cond_map, dose)

    # Total number of vaccinations per day
    df = df[["date_index", dose]].groupby(["date_index"], as_index=False).sum()

    # Cumulative vaccination and coverage
    df[f"cml_{dose})"] = df[dose].cumsum()
    df["cml_coverage"] = df[f"cml_{dose})"] / pop

    vac_dates = df.date_index.to_numpy()
    coverage_values = df.cml_coverage.to_numpy()
    avg_vals = np.array(apply_moving_average(coverage_values, 7)) + TINY_NUMBER
    return vac_dates, avg_vals



def get_pop(cluster, start_age, end_age, input_db):
    pop = input_db.query(
        "population",
        columns=["population"],
        conditions={
            "year": 2020,
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
