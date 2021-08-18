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
COVID_OWID = os.path.join(INPUT_DATA_PATH, "owid", "owid-covid-data.csv")
COVID_VNM_TARGETS = os.path.join(PROJECTS_PATH, "covid_19", "vietnam", "vietnam", "timeseries.json")
COVID_HCMC_TARGETS = os.path.join(
    PROJECTS_PATH, "covid_19", "vietnam", "ho_chi_minh_city", "timeseries.json"
)
COVID_HCMC_CASES = os.path.join(INPUT_DATA_PATH, "covid_vnm", "cases.csv")
COVID_HCMC_TEST = os.path.join(INPUT_DATA_PATH, "covid_vnm", "testing.csv")
COVID_HCMC_URL = "https://docs.google.com/spreadsheets/d/1CKCT9uOfKNuF4KNs8vKnr3y-hokkg9kv/export?format=xlsx&id=1CKCT9uOfKNuF4KNs8vKnr3y-hokkg9kv"

TARGETS_VNM = {
    "notifications": "new_cases",
    "infection_deaths": "new_deaths",
}

TARGETS_HCMC = {
    "notifications": "daily_reported_cases",
    "infection_deaths": "daily_death",
    "hospital_occupancy": "total_severe_cases",
}


def preprocess_vnm_data():
    df = pd.read_csv(COVID_OWID)
    df = df[df.iso_code == "VNM"]
    df.date = pd.to_datetime(
        df.date, errors="coerce", format="%Y-%m-%d", infer_datetime_format=False
    )
    df["date_index"] = (df.date - COVID_BASE_DATETIME).dt.days
    df = df[df.date <= pd.to_datetime("today")]

    return df


df = preprocess_vnm_data()


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


update_timeseries(TARGETS_VNM, df, COVID_VNM_TARGETS)

df_cases = pd.read_excel(COVID_HCMC_URL, sheet_name=["Reported cases"])["Reported cases"]
df_cases.rename(columns=lambda x: x.lower().strip().replace(" ", "_"), inplace=True)
df_cases.to_csv(COVID_HCMC_CASES)

df_testing = pd.read_excel(COVID_HCMC_URL, skiprows=[0, 2], sheet_name=["Testing"])["Testing"]
df_testing.rename(columns=lambda x: x.lower().strip().replace(" ", "_"), inplace=True)
df_testing.date = pd.to_datetime(
    df_testing["date"], errors="coerce", format="%Y-%m-%d", infer_datetime_format=False
)
df_testing["date_index"] = (df_testing.date - COVID_BASE_DATETIME).dt.days
df_testing.rename(columns={"sum.1": "daily_test"}, inplace=True)
df_testing["region"] = "Ho Chi Minh City"
df_testing = df_testing[["date", "date_index", "region", "daily_test"]][1:]
df_testing.to_csv(COVID_HCMC_TEST)

df_cases.date = pd.to_datetime(
    df_cases["date"], errors="coerce", format="%Y-%m-%d", infer_datetime_format=False
)
df_cases["date_index"] = (df_cases.date - COVID_BASE_DATETIME).dt.days


update_timeseries(TARGETS_HCMC, df_cases, COVID_HCMC_TARGETS)


df_testing.columns
