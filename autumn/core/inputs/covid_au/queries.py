from datetime import date, datetime

import numpy as np
import pandas as pd
from autumn.core.inputs.database import get_input_db
from autumn.core.utils.utils import apply_moving_average
from autumn.settings.constants import COVID_BASE_DATETIME

TINY_NUMBER = 1e-6
VACC_COVERAGE_START_AGES = (0, 5, 12, 16, 20, 30, 40, 50, 60, 70, 80, 85)
VACC_COVERAGE_END_AGES = (4, 11, 15, 19, 29, 39, 49, 59, 69, 79, 84, 89)


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


def get_nt_testing_numbers():
    """
    Returns number of tests administered in Northern Territory.
    """
    input_db = get_input_db()
    df = input_db.query("covid_nt", columns=["date_index", "total"])
    test_series = pd.Series(list(df["total"]), index=df["date_index"]).dropna()
    test_series += TINY_NUMBER
    return test_series


def get_nt_vac_coverage(
    pop_type: str = "Northern Territory", start_age: int = 0, end_age: int = 104, dose=1
) -> pd.Series:
    """
    Calculates the vaccination coverage for the Northern Territory by population type, dose and age brackets.

    Args:
        pop_type (str, optional): Pass either [Northern Territory | NT_ABORIGINAL] . Defaults to "Northern Territory".
        start_age (int, optional): Start age bracket. Defaults to 0.
        end_age (int, optional): End age bracket. Defaults to 104.
        dose (int, optional): Pass either [1|2|3|4]. Defaults to 1.

    Returns:
        pd.Series: A series of cumulative coverage by date index
    """
    input_db = get_input_db()

    cond_map = {
        "population": pop_type,
        "dose_number": dose,
        "start_age>": start_age,
        "end_age<": end_age,
    }

    pop = get_pop(pop_type, start_age, end_age, input_db)
    df = get_historical_vac_num(input_db, cond_map)

    # Total number of vaccinations per day
    df = df[["date_index", "doses"]].groupby(["date_index"], as_index=False).sum()

    # Cumulative vaccination and coverage
    df["cml_doses"] = df["doses"].cumsum()
    df["cml_coverage"] = df.cml_n / pop

    vac_dates = df.date_index.to_numpy()
    coverage_values = df.cml_coverage.to_numpy()
    avg_vals = np.array(apply_moving_average(coverage_values, 7)) + TINY_NUMBER
    return pd.Series(avg_vals, index=vac_dates)


def get_historical_vac_num(input_db, cond_map):
    df = input_db.query(
        "covid_nt_vac.secret",
        columns=["date_index", "doses", "start_age", "end_age"],
        conditions=cond_map,
    )

    return df


def get_pop(pop_type, start_age, end_age, input_db):
    pop = input_db.query(
        "population",
        columns=["population"],
        conditions={
            "year": 2020,
            "region": pop_type,
            "start_age>": start_age,
            "end_age<": end_age,
        },
    )
    pop = pop.population.sum()
    return pop
