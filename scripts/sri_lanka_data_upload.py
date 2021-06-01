"""
Script for loading LKA data into calibration targets and default.yml

"""
import json


import os
import pandas as pd
from datetime import datetime


import regex as re


from autumn.region import Region

from settings import APPS_PATH
from settings import INPUT_DATA_PATH

APPS_PATH


# start date to calculate time since Dec 31, 2019
COVID_BASE_DATETIME = datetime(2019, 12, 31, 0, 0, 0)
COVID_LKA_DATACSV = os.path.join(INPUT_DATA_PATH, "covid_lka", "data.csv")
COVID_LKA_TARGETS = os.path.join(APPS_PATH, "covid_19", "regions", "sri_lanka", "targets.json")

TARGETS = {
    "notifications": "New COVID19 cases reported",
    "infection_deaths": "COVID19 deaths",
    "icu_occupancy": "Occupied beds in ICUs",
    "hospital_occupancy": "Total hospitalised COVID19 cases",
    "testing": "PCR tests done",
}


def preprocess_lka_data():
    df = pd.read_csv(COVID_LKA_DATACSV)
    df.periodname = pd.to_datetime(
        df.periodname, errors="coerce", format="%Y-%m-%d", infer_datetime_format=False
    )
    df["date_index"] = (df.periodname - COVID_BASE_DATETIME).dt.days
    df = df[df.periodname <= pd.to_datetime("today")]

    return df


df = preprocess_lka_data()
file_path = COVID_LKA_TARGETS
with open(file_path, mode="r") as f:
    targets = json.load(f)
for key, val in TARGETS.items():
    # Drop the NaN value rows from df before writing data.
    temp_df = df[["date_index", val]].dropna(0, subset=[val])

    targets[key]["times"] = list(temp_df["date_index"])
    targets[key]["values"] = list(temp_df[val])
with open(file_path, "w") as f:
    json.dump(targets, f, indent=2)

