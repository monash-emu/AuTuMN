"""
Script for loading IDN data into calibration targets and default.yml

"""
import json


import os
import pandas as pd
from autumn.settings import PROJECTS_PATH
from autumn.settings import INPUT_DATA_PATH

from autumn.models.covid_19.constants import COVID_BASE_DATETIME

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
    df.date = pd.to_datetime(
        df.date, errors="coerce", format="%Y-%m-%d", infer_datetime_format=False
    )
    df["date_index"] = (df.date - COVID_BASE_DATETIME).dt.days
    df = df[df.date <= pd.to_datetime("today")]

    return df


def preprocess_bali_data():
    df = pd.read_excel(COVID_BALI_DATA, sheet_name=0)
    df.rename(columns=lambda x: x.lower().strip().replace(" ", "_"), inplace=True)
    df.date = pd.to_datetime(
        df.date, errors="coerce", format="%Y-%m-%d", infer_datetime_format=False
    )
    df["date_index"] = (df.date - COVID_BASE_DATETIME).dt.days

    return df


def update_timeseries(TARGETS, df, file_path):
    with open(file_path, mode="r") as f:
        targets = json.load(f)
    for key, val in TARGETS.items():
        # Drop the NaN value rows from df before writing data.
        temp_df = df[["date_index", val]].dropna(0, subset=[val])

        targets[key]["times"] = list(temp_df["date_index"])
        targets[key]["values"] = list(temp_df[val])
    with open(file_path, "w") as f:
        json.dump(targets, f, indent=2)


df = preprocess_idn_data()
update_timeseries(TARGETS_IDN, df, COVID_IDN_TARGETS)

df = preprocess_bali_data()
update_timeseries(TARGETS_BALI, df, COVID_BALI_TARGETS)
