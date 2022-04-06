"""
Script for loading LKA data into calibration targets and default.yml

"""
import json

import os
import pandas as pd
from datetime import datetime

from autumn.settings import PROJECTS_PATH
from autumn.settings import INPUT_DATA_PATH
from autumn.tools.utils.utils import update_timeseries
from autumn.models.covid_19.constants import COVID_BASE_DATETIME
from autumn.tools.utils.utils import create_date_index


LKA_DATA_2021 = os.path.join(INPUT_DATA_PATH, "covid_lka", "data_2021.csv")
LKA_DATA_2022 = os.path.join(INPUT_DATA_PATH, "covid_lka", "data.csv")
COVID_LKA_REGION = {"sri_lanka": "Sri Lanka"}
COVID_LKA_TARGETS = os.path.join(
    PROJECTS_PATH, "covid_19", "sri_lanka", "sri_lanka", "timeseries.json"
)

TARGETS_MAP_LKA = {
    "notifications": "New COVID19 cases reported",
    "infection_deaths": "COVID19 deaths",
    "icu_occupancy": "Occupied beds in ICUs",
    "hospital_occupancy": "Total hospitalised COVID19 cases",
}


def preprocess_lka_data():

    df = pd.concat([pd.read_csv(file) for file in [LKA_DATA_2021, LKA_DATA_2022]])
    df.periodname = pd.to_datetime(
        df.periodname, errors="coerce", format="%Y-%m-%d", infer_datetime_format=False
    )
    df["date_index"] = (df.periodname - COVID_BASE_DATETIME).dt.days
    df = df[df.periodname <= pd.to_datetime("today")]
    # Fix errors - email sent to sl team
    df.loc[
        df.periodid == 20201231,
        [
            "Sri Lanka Total bed occupancy in COVID19  ICUs",
            "Sri Lanka Occupied beds in ICUs (Source: DPRD)",
        ],
    ] = [19.0, 19.0]
    df.loc[
        df.periodid == 20210902,
        [
            "Sri Lanka Total bed occupancy in COVID19  ICUs",
            "Sri Lanka Occupied beds in ICUs (Source: DPRD)",
        ],
    ] = [100, 100]

    return df


df = preprocess_lka_data()

for region, col_name in COVID_LKA_REGION.items():
    file_path = os.path.join(PROJECTS_PATH, "covid_19", "sri_lanka", region, "timeseries.json")

    region_select = [each_col for each_col in df.columns if col_name in each_col]
    region_df = df[["date_index"] + region_select]
    with open(file_path, mode="r") as f:
        targets = json.load(f)
    for key, val in TARGETS_MAP_LKA.items():
        # Drop the NaN value rows from df before writing data.
        col_select = [each_col for each_col in region_df.columns if val in each_col]
        col_select = (
            col_select[1] if region == "sri_lanka_wp" and key == "notifications" else col_select[0]
        )
        temp_df = region_df[["date_index", col_select]].dropna(0, subset=[col_select])

        targets[key]["times"] = list(temp_df["date_index"])
        targets[key]["values"] = list(temp_df[col_select])
    with open(file_path, "w") as f:
        json.dump(targets, f, indent=2)
