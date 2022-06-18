from re import sub
from typing import Tuple, Optional
import pandas as pd
import os
from datetime import datetime

from autumn.settings import COVID_BASE_AGEGROUPS
from autumn.core.inputs.covid_au.queries import get_vic_testing_numbers
from autumn.core.inputs.covid_phl.queries import get_phl_subregion_testing_numbers
from autumn.core.inputs.covid_lka.queries import get_lka_testing_numbers
from autumn.core.inputs.covid_mmr.queries import get_mmr_testing_numbers
from autumn.core.inputs.covid_bgd.queries import get_coxs_bazar_testing_numbers
from autumn.core.inputs.owid.queries import get_international_owid_numbers
from autumn.core.inputs.covid_btn.queries import get_btn_testing_numbers
from autumn.settings.folders import INPUT_DATA_PATH
from autumn.settings.constants import COVID_BASE_DATETIME

base_dir = os.path.dirname(os.path.abspath(os.curdir))
EUR_TESTING_FOLDER = os.path.join(INPUT_DATA_PATH, "testing")


def get_uk_testing_numbers():
    data_path = os.path.join(EUR_TESTING_FOLDER, "UK", "data_2021-May-17.csv")
    data = pd.read_csv(data_path)
    data = data.dropna()
    date_str_to_int = lambda s: (datetime.strptime(s, "%Y-%m-%d") - COVID_BASE_DATETIME).days
    test_dates = list(data.date.apply(date_str_to_int).to_numpy())
    test_numbers = list(data.loc[:, "newVirusTests"])

    # reverse order
    test_dates = test_dates[::-1]
    test_numbers = test_numbers[::-1]

    return test_dates, test_numbers


def get_eu_testing_numbers(iso3):

    iso3_map = {"BEL": "Belgium", "FRA": "France", "ITA": "Italy", "ESP": "Spain", "SWE": "Sweden"}

    data_path = os.path.join(EUR_TESTING_FOLDER, "EU", "download_18May2021.csv")
    data = pd.read_csv(data_path)

    mask_country = data.country == iso3_map[iso3]
    country_data = data[mask_country]

    mask_national = country_data.level == "national"
    national_data = country_data[mask_national]

    year_week = list(national_data.year_week)
    n_tests_per_week = list(national_data.tests_done)

    test_dates, test_numbers = convert_weekly_total_to_daily_numbers(year_week, n_tests_per_week)

    return test_dates, test_numbers


def convert_weekly_total_to_daily_numbers(year_week, n_tests_per_week):
    test_dates = []
    test_numbers = []

    for i_week in range(len(year_week)):
        y_w = year_week[i_week]
        week_val = n_tests_per_week[i_week]

        # convert year week to integer
        year = y_w.split("-")[0]
        week = y_w.split("W")[1]

        # work out out date_int, which is the integer representing the first day of the selected week
        ref_integer = -8 if year == "2020" else 363
        date_int = ref_integer + 7 * float(week)

        test_dates += [date_int + i for i in range(7)]
        test_numbers += [week_val / 7.0 for _ in range(7)]

    return test_dates, test_numbers


def get_testing_numbers_for_region(
    country_iso3: str, 
    subregion: Optional[str]
) -> Tuple[list, list]:
    """
    Use the appropriate function to retrieve the testing numbers applicable to the region being modelled.
    Functions are taken from the autumn input tools module, as above.

    Args:
        country_iso3: ISO3 code for the country being simulated
        subregion: Name of the country's subregion being considered or None if it is a national level
    Return:
        The testing data for the country
    """

    subregion = subregion or False

    if country_iso3 == "PHL":
        phl_region = subregion.lower() if subregion else "philippines"
        test_dates, test_values = get_phl_subregion_testing_numbers(phl_region)
    elif country_iso3 == "GBR":
        test_dates, test_values = get_uk_testing_numbers()
    elif country_iso3 in ("BEL", "ITA", "SWE", "FRA", "ESP"):
        test_dates, test_values = get_eu_testing_numbers(country_iso3)
    elif country_iso3 == "LKA":
        test_dates, test_values = get_lka_testing_numbers()
    elif country_iso3 == "MMR":
        test_df = get_mmr_testing_numbers()
        msg = "Negative test values present"
        assert (test_df >= 0).all()
        return test_df
    elif country_iso3 == "BGD" and subregion == "FDMN":
        test_df = get_coxs_bazar_testing_numbers()
        msg = "Negative test values present"
        assert (test_df >= 0).all()
        return test_df
    elif country_iso3 == "BTN":
        test_df = get_btn_testing_numbers(subregion)
        msg = "Negative test values present"
        assert (test_df >= 0).all()
        return test_df
    else:
        test_df = get_international_owid_numbers(country_iso3)
        msg = "Negative test values present"
        assert (test_df >= 0).all()
        return test_df
        
    # Check data and return
    msg = "Length of test dates and test values are not equal"
    assert len(test_dates) == len(test_values), msg
    test_df = pd.Series(test_values, index=test_dates)
    msg = "Negative test values present"
    assert (test_df >= 0).all()
    return test_df
