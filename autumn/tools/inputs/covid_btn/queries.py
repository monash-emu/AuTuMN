from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd

from autumn.tools.inputs.database import get_input_db
from autumn.tools.inputs.demography.queries import get_population_by_agegroup
from autumn.tools.utils.utils import check_list_increasing

TINY_NUMBER = 1e-6


def get_btn_testing_numbers(subregion: Optional[str]):
    """
    Returns number of tests administered in Bhutan or Thimphu.
    """

    subregion = "Bhutan" if subregion is False else subregion

    cond_map = {
        "region": subregion,
    }

    input_db = get_input_db()
    df = input_db.query(
        "covid_btn_test", columns=["date_index", "total_tests"], conditions=cond_map
    )
    df.dropna(inplace=True)
    test_dates = df.date_index.to_numpy()
    values = df["total_tests"].to_numpy() + TINY_NUMBER
    values = values / 6.5

    return test_dates, values


def get_btn_vac_coverage(
    region: str,
    dose: int,
) -> pd.Series:
    """Calculates the vaccination coverage for Bhutan and Thimphu

    Args:
        region (str): Can be {"Bhutan"|"Thimphu"}.
        dose (int): Can be {1|2|3}.

    Returns:
        pd.Series: A Pandas series of dates and coverage values
    """

    # Get the total population
    pop_region = {"Bhutan": None, "Thimphu": "Thimphu"}
    population = get_population_by_agegroup([0], "BTN", pop_region[region], 2022)

    input_db = get_input_db()

    cond_map = {
        "dose_num": str(dose),
        "region": region,
    }

    df = input_db.query(
        "covid_btn_vac", columns=["date_index", "num"], conditions=cond_map
    )
    df = df.groupby("date_index", as_index=False).sum()

    # Calculate the coverage
    df["coverage"] = df["num"] / population

    vac_dates = df["date_index"].to_numpy()
    vac_coverage = df["coverage"].to_numpy()

    coverage_too_large = any(vac_coverage >= 0.99)
    not_increasing_coverage = check_list_increasing(vac_coverage)

    # Validation
    if any([coverage_too_large, not_increasing_coverage]):
        AssertionError("Unrealistic coverage")

    return pd.Series(vac_coverage, index=vac_dates)
