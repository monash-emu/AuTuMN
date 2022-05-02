"""
Script for loading NPL data into calibration targets and default.yml

"""
import json


import os
import pandas as pd
from datetime import datetime


from autumn.settings import PROJECTS_PATH
from autumn.settings import INPUT_DATA_PATH
from autumn.models.covid_19.constants import COVID_BASE_DATETIME

# Use OWID csv for notification and death numbers.
COVID_NPL_OWID = os.path.join(INPUT_DATA_PATH, "owid", "owid-covid-data.csv")
COVID_NPL_TARGETS = os.path.join(PROJECTS_PATH, "covid_19", "nepal", "timeseries.json")
COVID_NPL_DATA = os.path.join(INPUT_DATA_PATH, "covid_npl", "cases.csv")


TARGETS = {
    "notifications": "new_case_npl_data",
    "infection_deaths": "new_deaths",
}


def preprocess_npl_data():
    df_owid = pd.read_csv(COVID_NPL_OWID)
    df_owid = df_owid[df_owid.iso_code == "NPL"]
    df_owid.date = pd.to_datetime(
        df_owid.date, errors="coerce", format="%Y-%m-%d", infer_datetime_format=False
    )
    df_owid = df_owid[df_owid.date <= pd.to_datetime("today")]

    df_npl = pd.read_csv(COVID_NPL_DATA, index_col=0)
    df_npl.date = pd.to_datetime(
        df_npl.date, errors="coerce", format="%Y-%m-%d", infer_datetime_format=False
    )

    df_npl = df_npl.groupby(["date"]).count()
    df_npl.rename(columns={"age": "new_case_npl_data"}, inplace=True)
    df_owid = df_owid.merge(df_npl, how="outer", on=["date"])

    df_owid["date_index"] = (df_owid.date - COVID_BASE_DATETIME).dt.days
    return df_owid


df = preprocess_npl_data()
file_path = COVID_NPL_TARGETS
with open(file_path, mode="r") as f:
    targets = json.load(f)
for key, val in TARGETS.items():
    # Drop the NaN value rows from df before writing da ta.
    temp_df = df[["date_index", val]].dropna(0, subset=[val])
    temp_df = temp_df.sort_values(by="date_index")
    targets[key]["times"] = list(temp_df["date_index"])
    targets[key]["values"] = list(temp_df[val])
with open(file_path, "w") as f:
    json.dump(targets, f, indent=2)
