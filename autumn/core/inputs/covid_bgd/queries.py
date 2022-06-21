from datetime import date, datetime

import numpy as np
import pandas as pd

from autumn.core.inputs.database import get_input_db
from autumn.core.inputs.demography.queries import get_population_by_agegroup
from autumn.core.utils.utils import check_list_increasing

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


def get_bgd_vac_coverage(region: str, vaccine: str, dose: int):
    """Calculates the vaccination coverage for Bangladesh and sub regions

    Args:
        region (str, required): Can be {"BGD"|"DHK"}.
        vaccine (str, required): Can be {"astrazeneca"|"pfizer"|"sinopharm"|"moderna"|"sinovac"|"total"}.
        dose (int, required): Can be {1|2}.

    Returns:
        (Pandas series): A Pandas series of dates(index) and coverage(values)
    """

    # Get the total population
    pop_region = {"BGD": None, "DHK": "Dhaka district"}
    population = get_population_by_agegroup([0], "BGD", pop_region[region], 2021)

    input_db = get_input_db()

    cond_map = {
        "dose": str(dose),
        "region": region,
    }

    df = input_db.query(
        "bgd_vacc", columns=["date_index", vaccine], conditions=cond_map
    )

    # Calculate the coverage
    df["coverage"] = df[vaccine] / population

    vac_dates = df["date_index"].to_numpy()
    vac_coverage = df["coverage"].to_numpy()

    coverage_too_large = any(vac_coverage >= 0.99)
    not_increasing_coverage = check_list_increasing(vac_coverage)

    # Validation
    if any([coverage_too_large, not_increasing_coverage]):
        AssertionError("Unrealistic coverage")

    return pd.Series(vac_coverage, index=vac_dates)
