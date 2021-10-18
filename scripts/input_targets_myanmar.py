"""
Script for loading Myanmar data into calibration targets and default.yml
"""

import json
import os
import pandas as pd
from datetime import datetime

from autumn.settings import PROJECTS_PATH
from autumn.settings import INPUT_DATA_PATH
from autumn.tools.utils.utils import update_timeseries
from autumn.tools.utils.utils import create_date_index

# start date to calculate time since Dec 31, 2019
COVID_BASE_DATETIME = datetime(2019, 12, 31, 0, 0, 0)

# Use OWID csv for notification and death numbers.
COVID_MMR_OWID = os.path.join(INPUT_DATA_PATH, "owid", "owid-covid-data.csv")
COVID_MMR_TARGETS = os.path.join(PROJECTS_PATH, "covid_19", "myanmar", "timeseries.json")
COVID_MMR_DATA = os.path.join(INPUT_DATA_PATH, "covid_mmr", "cases.csv")

URL = "https://docs.google.com/spreadsheets/d/1VeUof9_-s0bsndo8tLsCwnAhkUUZgsdV-r980gumMPA/export?format=csv&id=1VeUof9_-s0bsndo8tLsCwnAhkUUZgsdV-r980gumMPA"

TARGETS = {
    "notifications": "cases",
    "infection_deaths": "deaths",
}

mmr_df = pd.read_csv(URL)
mmr_df.to_csv(COVID_MMR_DATA)

str_col = ["Tests", "Cases", "Recovered", "Negative"]

mmr_df[str_col] = mmr_df[str_col].replace(to_replace=r',', value='', regex=True)
mmr_df[str_col] = mmr_df[str_col].apply(pd.to_numeric)
mmr_df["Date"] = pd.to_datetime(mmr_df['Date'])
mmr_df = create_date_index(COVID_BASE_DATETIME, mmr_df, "Date")


update_timeseries(TARGETS,mmr_df,COVID_MMR_TARGETS)



def preprocess_npl_data():
    df_owid = pd.read_csv(COVID_MMR_OWID)
    df_owid = df_owid[df_owid.iso_code == "NPL"]
    df_owid.date = pd.to_datetime(
        df_owid.date, errors="coerce", format="%Y-%m-%d", infer_datetime_format=False
    )
    df_owid = df_owid[df_owid.date <= pd.to_datetime("today")]

    df_npl = pd.read_csv(COVID_MMR_DATA, index_col=0)
    df_npl.date = pd.to_datetime(
        df_npl.date, errors="coerce", format="%Y-%m-%d", infer_datetime_format=False
    )

    df_npl = df_npl.groupby(["date"]).count()
    df_npl.rename(columns={"age": "new_case_npl_data"}, inplace=True)
    df_owid = df_owid.merge(df_npl, how="outer", on=["date"])

    df_owid["date_index"] = (df_owid.date - COVID_BASE_DATETIME).dt.days
    return df_owid


df = preprocess_npl_data()
file_path = COVID_MMR_TARGETS
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
