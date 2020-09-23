#!/usr/bin/env python
"""
Script for loading MYS data into calibration targets and default.yml

Manually download file "Total Death Covid19 in Malaysia_updated ..."
from https://drive.google.com/drive/u/1/folders/1C-OsJbjdhQmpmedBZ8XuM65ANBO22dXc
to ./data/inputs/covid_mys/COVID_MYS_DEATH.csv for now!
"""
import os
import sys
import json
from datetime import datetime
from google_drive_downloader import GoogleDriveDownloader as gdd
from autumn import constants
import pandas as pd


# From WHO google drive folder
CASE_DATA_URL = "1mnZcmj2jfmrap1ytyg_ErD1DZ7zDptyJ"  # shareable link
# DEATH_DATA_URL = "1cQe1k7GQRKFzcfXXxdL7_pTNQUMpOAPX3PqhPsI4xt8"
COVID_MYS_DIRPATH = os.path.join(constants.INPUT_DATA_PATH, "covid_mys")
COVID_MYS_CASE_PATH = os.path.join(COVID_MYS_DIRPATH, "COVID_MYS_CASE.xlsx")
COVID_MYS_DEATH_PATH = os.path.join(COVID_MYS_DIRPATH, "COVID_MYS_DEATH.csv")

gdd.download_file_from_google_drive(
    file_id=CASE_DATA_URL, dest_path=COVID_MYS_CASE_PATH, overwrite=True
)
# gdd.download_file_from_google_drive(file_id = DEATH_DATA_URL, dest_path = COVID_MYS_DEATH_PATH,overwrite=True)

COVID_BASE_DATE = pd.datetime(2019, 12, 31)
REGION_DIR = os.path.join(constants.APPS_PATH, "covid_19", "regions", "malaysia")

TARGETS_MAP = {
    "notifications": "NC",
    "icu_occupancy": "ICU",
    "infection_deaths": "Death per day",
}


def main():

    update_calibration()


def update_calibration():
    """
    Update Malaysia's calibration targets
    """

    df = load_mys_data()

    file_path = os.path.join(REGION_DIR, "targets.json")
    with open(file_path, mode="r") as f:
        targets = json.load(f)
    for key, val in TARGETS_MAP.items():
        # Drop the NaN value rows from df before writing data.
        temp_df = df[["date_index", val]].dropna(0, subset=[val])

        targets[key]["times"] = list(temp_df["date_index"])
        targets[key]["values"] = list(temp_df[val])
    with open(file_path, "w") as f:
        json.dump(targets, f, indent=2)


def load_mys_data():

    case_df = pd.read_excel(COVID_MYS_CASE_PATH)
    death_df = pd.read_csv(COVID_MYS_DEATH_PATH)

    case_df.Date = pd.to_datetime(
        case_df["Date"],
        errors="coerce",
        format="%Y-%m-%d %H:%M:%S",
        infer_datetime_format=True,
    )
    case_df = case_df[(case_df.Date > "2020-03-02") & (case_df.Date < "2020-12-31")]
    case_df["date_index"] = (case_df.Date - COVID_BASE_DATE).dt.days
    case_df.rename(
        columns={
            "New Cases (A)": "NC",
            "Imported cases (B)": "IC",
            "Total ICU Usage including ventilator usage (E)": "ICU",
        },
        inplace=True,
    )
    case_df.drop(
        columns=[
            "Active cases (hospitalised) including ICU (C)",
            "Active cases exclude ICU (D = C - E)",
            "Ventilator Usage (F)",
            "Unnamed: 7",
        ],
        inplace=True,
    )

    for column in death_df.columns:
        if column not in {"Date", "Death per day"}:
            death_df.drop(columns=[column], inplace=True)
    death_df.Date = death_df.Date.str[:] + "-2020"
    death_df.Date = pd.to_datetime(
        death_df["Date"], errors="coerce", format="%d-%b-%Y", infer_datetime_format=True
    )
    death_df["date_index"] = (death_df.Date - COVID_BASE_DATE).dt.days
    death_df = death_df[death_df.Date.notnull() & death_df["Death per day"].notnull()]

    df = pd.merge(
        case_df, death_df, how="left", left_on=["date_index"], right_on=["date_index"]
    )
    df.drop(columns=["Date_y"], inplace=True)

    return df


if __name__ == "__main__":
    main()
