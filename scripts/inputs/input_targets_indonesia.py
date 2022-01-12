"""
Script for loading IDN data into calibration targets and default.yml

"""
import json


import os
import pandas as pd
from datetime import datetime
from autumn.tools.utils.utils import create_date_index, update_timeseries
from autumn.settings import PROJECTS_PATH
from autumn.settings import INPUT_DATA_PATH

# start date to calculate time since Dec 31, 2019
COVID_BASE_DATETIME = datetime(2019, 12, 31, 0, 0, 0)

# Use OWID csv for notification and death numbers.
COVID_IDN_OWID = os.path.join(INPUT_DATA_PATH, "owid", "owid-covid-data.csv")
COVID_IDN_DATA = os.path.join(INPUT_DATA_PATH, "covid_idn", "cases_by_province.xlsx")
COVID_IDN_TARGETS = os.path.join(
    PROJECTS_PATH, "covid_19", "indonesia", "indonesia", "timeseries.json"
)

COVID_BALI_DATA = os.path.join(INPUT_DATA_PATH, "covid_idn", "Bali Modelling.xlsx")
COVID_BALI_TARGETS = os.path.join(PROJECTS_PATH, "covid_19", "indonesia", "bali", "timeseries.json")

TARGETS_IDN = {
    "notifications": "new_cases",
    "infection_deaths": "new_deaths",
}

TARGETS_BALI = {
    "notifications": "daily_confirmed",
    "infection_deaths": "death_daily",
}


def preprocess_idn_data():
    df = pd.read_csv(COVID_IDN_OWID)
    df = df[df.iso_code == "IDN"]
    df = create_date_index(COVID_BASE_DATETIME, df, "date")
    df = df[df.date <= pd.to_datetime("today")]

    return df


def preprocess_bali_data():
    df = pd.read_excel(COVID_BALI_DATA, sheet_name=0)
    df = create_date_index(COVID_BASE_DATETIME, df, "date")

    return df


df = preprocess_idn_data()
update_timeseries(TARGETS_IDN, df, COVID_IDN_TARGETS)

df = preprocess_bali_data()
update_timeseries(TARGETS_BALI, df, COVID_BALI_TARGETS)
