"""
Script for loading NPL data into calibration targets and default.yml

"""
import json


import os
import pandas as pd
from datetime import datetime


from autumn.settings import PROJECTS_PATH
from autumn.settings import INPUT_DATA_PATH


# start date to calculate time since Dec 31, 2019
COVID_BASE_DATETIME = datetime(2019, 12, 31, 0, 0, 0)

# Use OWID csv for notification and death numbers.
COVID_VNM_DATACSV = os.path.join(INPUT_DATA_PATH, "owid", "owid-covid-data.csv")
COVID_VNM_TARGETS = os.path.join(PROJECTS_PATH, "covid_19", "vietnam","vietnam", "timeseries.json")

TARGETS = {
    "notifications": "new_cases",
    "infection_deaths": "new_deaths",
}


def preprocess_vnm_data():
    df = pd.read_csv(COVID_VNM_DATACSV)
    df = df[df.iso_code == "VNM"]
    df.date = pd.to_datetime(
        df.date, errors="coerce", format="%Y-%m-%d", infer_datetime_format=False
    )
    df["date_index"] = (df.date - COVID_BASE_DATETIME).dt.days
    df = df[df.date <= pd.to_datetime("today")]

    return df


df = preprocess_vnm_data()
file_path = COVID_VNM_TARGETS
with open(file_path, mode="r") as f:
    targets = json.load(f)
for key, val in TARGETS.items():
    # Drop the NaN value rows from df before writing data.
    temp_df = df[["date_index", val]].dropna(0, subset=[val])

    targets[key]["times"] = list(temp_df["date_index"])
    targets[key]["values"] = list(temp_df[val])
with open(file_path, "w") as f:
    json.dump(targets, f, indent=2)
