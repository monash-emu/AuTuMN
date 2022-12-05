from typing import List

import numpy as np
import pandas as pd
from autumn.core.inputs.database import get_input_db
from autumn.core.inputs.demography.queries import get_population_by_agegroup
from autumn.core.utils.utils import apply_moving_average, check_list_increasing


def get_phl_subregion_testing_numbers(region):
    """
    Returns 7-day moving average of number of tests administered in Philippines & sub regions.
    """

    # A hack to revet back to national testing for the sub regions.
    region = "philippines" if region in {"western visayas", "barmm"} else region

    input_db = get_input_db()
    df = input_db.query(
        "covid_phl",
        columns=["date_index", "daily_output_unique_individuals"],
        conditions={"facility_name": region},
    )
    test_dates = df.date_index.to_numpy()
    test_values = df.daily_output_unique_individuals.to_numpy()
    epsilon = 1e-6  # A really tiny number to avoid having any zeros
    avg_vals = np.array(apply_moving_average(test_values, 7)) + epsilon
    return pd.Series(avg_vals, index=test_dates)


def get_phl_vac_coverage(
    dose: str, region: str = "NATIONAL CAPITAL REGION (NCR)", vaccine: List[str] = None
) -> pd.Series:
    """Calculates the vaccination coverage for any sub region of the Philippines.

    Args:
        dose (str): Can be any {FIRST_DOSE | SECOND_DOSE | SINGLE_DOSE | BOOSTER_DOSE | ADDITIONAL_DOSE}
        region (str, optional): Can be any defined 'REGION' of Philippines. Defaults to "NATIONAL CAPITAL REGION (NCR)".
        vaccine (List[str], optional): Can be { aztrazeneca | gamaleya+1 | gamaleya+2 |janssen | moderna | pfizer
         | sinopharm | sinovac+coronavac | sputnik+light}. Defaults to None, implies all.

    Returns:
        pd.Series: A Pandas series of dates(index) and coverage(values)
    """

    # Get the total population
    pop_region = {"NATIONAL CAPITAL REGION (NCR)": "Metro Manila"}
    population = get_population_by_agegroup([0], "PHL", pop_region[region], 2020)

    if vaccine is not None:
        vaccine = [vac.lower() for vac in vaccine]
        valid = {
            "aztrazeneca",
            "gamaleya+1",
            "gamaleya+2",
            "janssen",
            "moderna",
            "pfizer",
            "sinopharm",
            "sinovac+coronavac",
            "sputnik+light",
        }

        assert all(vac in valid for vac in vaccine), "Invalid vaccine name"

    input_db = get_input_db()

    cond_map = {
        "cml_dose": dose,
    }

    df = input_db.query(
        "covid_phl_vac",
        columns=["date_index", "vaccination", f"{region}"],
        conditions=cond_map,
    )

    if vaccine is not None:
        df = df[df["vaccination"].isin(vaccine)]

    df = df.groupby("date_index", as_index=False).sum()

    # Calculate the coverage
    df["coverage"] = df[region] / population

    vac_dates = df["date_index"].to_numpy()
    vac_coverage = df["coverage"].to_numpy()

    coverage_too_large = any(vac_coverage >= 0.99)
    not_increasing_coverage = check_list_increasing(vac_coverage)

    # Validation
    if any([coverage_too_large, not_increasing_coverage]):
        AssertionError("Unrealistic coverage")

    return pd.Series(vac_coverage, index=vac_dates)
