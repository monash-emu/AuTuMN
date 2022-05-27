from typing import List
import numpy as np
import pandas as pd

from autumn.core.inputs.database import get_input_db
from autumn.core.utils.utils import check_list_increasing
from autumn.core.inputs.demography.queries import get_population_by_agegroup

def get_mys_vac_coverage(
    dose: str,
) -> pd.Series:
    """Calculates the vaccination coverage for Malaysia.

    Args:
        dose (str): Can be any { partial | full | booster}

    Returns:
        pd.Series: A Pandas series of dates(index) and coverage(values)
    """

    # Get the total population
    population = get_population_by_agegroup([0], "MYS", None, 2020)

    input_db = get_input_db()

    cond_map = {
        "dose": dose,
    }

    df = input_db.query(
        "covid_mys_vac",
        columns=["date_index", "number"],
        conditions=cond_map,
    )

    df = df.groupby("date_index", as_index=False).sum()
    df["cml_number"] = df["number"].cumsum()
    # Calculate the coverage
    df["coverage"] = round(df["cml_number"] / population,3)

    vac_dates = df["date_index"].to_numpy()
    vac_coverage = df["coverage"].to_numpy()

    coverage_too_large = any(vac_coverage >= 0.99)
    not_increasing_coverage = check_list_increasing(vac_coverage)

    # Validation
    if any([coverage_too_large, not_increasing_coverage]):
        AssertionError("Unrealistic coverage")

    return pd.Series(vac_coverage, index=vac_dates)
