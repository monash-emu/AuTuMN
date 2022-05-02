import os
import pandas as pd
from datetime import datetime

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
