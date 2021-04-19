##!/usr/bin/env python
"""
Script for loading MYS data into calibration targets and default.yml

"""
import json
import os
import sys
from datetime import datetime

import pandas as pd
import regex as re
import yaml
from google_drive_downloader import GoogleDriveDownloader as gdd

import settings
from autumn.region import Region

# From WHO google drive folder
CASE_DATA_URL = "https://docs.google.com/spreadsheets/d/1FbYG8szgbvw3pjULWEHDZ8uhu5BooV_-cygLn7iLaxA/export?format=xlsx&id=1FbYG8szgbvw3pjULWEHDZ8uhu5BooV_-cygLn7iLaxA"  # shareable link
COVID_MYS_DEATH_URL = "https://docs.google.com/spreadsheets/d/1iv0veITNSKxpoVvY2WDOePgPC0LvFmOSjsYEq-5BsMM/export?format=xlsx&id=1iv0veITNSKxpoVvY2WDOePgPC0LvFmOSjsYEq-5BsMM"
COVID_REGIONAL_URL = "https://docs.google.com/spreadsheets/d/1ouWp2ge5zrVh1gCDPONN-TMaf1b7TqRr4f07UZLWSdg/export?format=xlsx&id=1ouWp2ge5zrVh1gCDPONN-TMaf1b7TqRr4f07UZLWSdg"

COVID_MYS_DIRPATH = os.path.join(settings.folders.INPUT_DATA_PATH, "covid_mys")
COVID_MYS_CASE_CSV = os.path.join(COVID_MYS_DIRPATH, "COVID_MYS_CASE.csv")
COVID_MYS_DEATH_CSV = os.path.join(COVID_MYS_DIRPATH, "COVID_MYS_DEATH.csv")
COVID_REGIONAL_CSV = os.path.join(COVID_MYS_DIRPATH, "COVID_REGIONAL.csv")


COVID_BASE_DATE = pd.datetime(2019, 12, 31)
REGION_MYS = os.path.join(settings.folders.APPS_PATH, "covid_19", "regions", "malaysia")
REGION_SABAH = os.path.join(settings.folders.APPS_PATH, "covid_19", "regions", "sabah")
REGION_SELANGOR = os.path.join(settings.folders.APPS_PATH, "covid_19", "regions", "selangor")
REGION_JOHOR = os.path.join(settings.folders.APPS_PATH, "covid_19", "regions", "johor")
REGION_KUALA_LUMPUR = os.path.join(
    settings.folders.APPS_PATH, "covid_19", "regions", "kuala_lumpur"
)
REGION_PENANG = os.path.join(settings.folders.APPS_PATH, "covid_19", "regions", "penang")

TARGETS_MYS = {
    "notifications": "NC",
    "icu_occupancy": "ICU",
    "infection_deaths": "Death per day",
}


TARGETS_SABAH = {"notifications": "local_cases", "infection_deaths": "sabah_death"}
TARGETS_SELANGOR = {"notifications": "local_cases", "infection_deaths": "selangor_death"}
TARGETS_JOHOR = {"notifications": "local_cases", "infection_deaths": "johor_death"}
TARGETS_KUALA_LUMPUR = {"notifications": "local_cases", "infection_deaths": "kuala_lumpur_death"}
TARGETS_PENANG = {"notifications": "local_cases", "infection_deaths": "penang_death"}


def main():

    update_calibration(Region.MALAYSIA)
    update_calibration(Region.SABAH)
    update_calibration(Region.SELANGOR)
    update_calibration(Region.JOHOR)
    update_calibration(Region.PENANG)
    update_calibration(Region.KUALA_LUMPUR)


def update_calibration(REGION: str):
    """
    Update calibration targets
    """

    df = load_data(REGION)
    update_target(REGION, df)


def update_target(REGION, df):

    if REGION == "malaysia":
        TARGET = TARGETS_MYS
        REGION = REGION_MYS
    elif REGION == "sabah":
        TARGET = TARGETS_SABAH
        REGION = REGION_SABAH
    elif REGION == "selangor":
        TARGET = TARGETS_SELANGOR
        REGION = REGION_SELANGOR
    elif REGION == "johor":
        TARGET = TARGETS_JOHOR
        REGION = REGION_JOHOR
    elif REGION == "penang":
        TARGET = TARGETS_PENANG
        REGION = REGION_PENANG
    elif REGION == "kuala-lumpur":
        TARGET = TARGETS_KUALA_LUMPUR
        REGION = REGION_KUALA_LUMPUR

    file_path = os.path.join(REGION, "targets.json")
    with open(file_path, mode="r") as f:
        targets = json.load(f)
    for key, val in TARGET.items():
        # Drop the NaN value rows from df before writing data.
        temp_df = df[["date_index", val]].dropna(0, subset=[val])

        targets[key]["times"] = list(temp_df["date_index"])
        targets[key]["values"] = list(temp_df[val])
    with open(file_path, "w") as f:
        json.dump(targets, f, indent=2)

    with open(REGION + "\\params\\default.yml", "r") as f:
        tmp = yaml.load(f, Loader=yaml.FullLoader)

    if tmp.get("Importation") is not None:

        tmp = {key: val for key, val in tmp.items() if key == "importation"}

        temp_df = df[["date_index", "IC"]].dropna(0, subset=["IC"])
        temp_df = temp_df[temp_df.date_index != 94]
        tmp["importation"]["case_timeseries"]["times"] = list(temp_df["date_index"])
        tmp["importation"]["case_timeseries"]["values"] = list(temp_df["IC"])

        with open(REGION + "\\params\\default.yml", "r") as f:
            default = f.readlines()
        end = default.index("  case_timeseries:\n")
        default = default[:end]

        with open(REGION + "\\params\\default.yml", "w") as f:
            yaml.dump(tmp, f)
        with open(REGION + "\\params\\default.yml", "r") as f:
            append = f.readlines()

        end = append.index("  movement_prop: null\n")
        default = default + append[1:end]

        with open(REGION + "\\params\\default.yml", "w") as f:
            f.writelines(default)


def load_data(REGION: str):

    death_df = get_death()
    if REGION == "malaysia":
        case_df = load_mys()
    elif REGION == "sabah":
        case_df = load_regional_cases("Sabah")
        death_df = death_df[["sabah_death", "date_index"]]
    elif REGION == "selangor":
        case_df = load_regional_cases("Selangor")
        death_df = death_df[["selangor_death", "date_index"]]
    elif REGION == "johor":
        case_df = load_regional_cases("Johor")
        death_df = death_df[["johor_death", "date_index"]]
    elif REGION == "penang":
        case_df = load_regional_cases("Pulau Pinang")
        death_df = death_df[["penang_death", "date_index"]]
    elif REGION == "kuala-lumpur":
        case_df = load_regional_cases("Kuala Lumpur")
        death_df = death_df[["kuala_lumpur_death", "date_index"]]

    df = pd.merge(case_df, death_df, how="outer", left_on=["date_index"], right_on=["date_index"])
    df.sort_values(by="date_index", inplace=True)
    return df


def get_death():
    death_df = pd.read_excel(COVID_MYS_DEATH_URL)
    death_df.to_csv(COVID_MYS_DEATH_CSV)
    for column in death_df.columns:
        if column not in {"Date", "Death per day", "state"}:
            death_df.drop(columns=[column], inplace=True)
    # death_df.Date = death_df.Date.str[:] + "-2020"
    death_df.Date = pd.to_datetime(
        death_df["Date"], errors="coerce", format="%Y-%m-%d", infer_datetime_format=False
    )
    death_df["date_index"] = (death_df.Date - COVID_BASE_DATE).dt.days
    death_df = death_df[death_df.Date.notnull() & death_df["Death per day"].notnull()]
    death_df.loc[death_df.state.isna(), "state"] = ""

    def fix_regex(each_row, state):

        x = re.findall(state + r"=(\d+)", each_row)
        if len(x) > 0:
            return int(x[0])
        else:
            return 0

    death_df["sabah_death"] = [fix_regex(each, "Sabah") for each in death_df.state]
    death_df["selangor_death"] = [fix_regex(each, "Selangor") for each in death_df.state]
    death_df["johor_death"] = [fix_regex(each, "Johor") for each in death_df.state]
    death_df["penang_death"] = [fix_regex(each, "Pulau Pinang") for each in death_df.state]
    death_df["kuala_lumpur_death"] = [fix_regex(each, "KL") for each in death_df.state]

    death_df.loc[death_df.state == "Sabah", "sabah_death"] = death_df["Death per day"]
    death_df.loc[death_df.state == "Selangor", "selangor_death"] = death_df["Death per day"]
    death_df.loc[death_df.state == "Johor", "johor_death"] = death_df["Death per day"]
    death_df.loc[death_df.state == "Penang", "penang_death"] = death_df["Death per day"]
    death_df.loc[death_df.state == "Kuala Lumpur", "kuala_lumpur_death"] = death_df["Death per day"]
    return death_df[
        [
            "date_index",
            "Death per day",
            "state",
            "sabah_death",
            "selangor_death",
            "johor_death",
            "penang_death",
            "kuala_lumpur_death",
        ]
    ]


def load_mys():
    case_df = pd.read_excel(CASE_DATA_URL)
    case_df.to_csv(COVID_MYS_CASE_CSV)

    case_df.Date = pd.to_datetime(
        case_df["Date"],
        errors="coerce",
        format="%Y-%m-%d %H:%M:%S",
        infer_datetime_format=True,
    )
    case_df = case_df[(case_df.Date > "2020-03-02") & (case_df.Date < "2021-12-31")]
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
            "Active cases including ICU (C)",
            "Active cases exclude ICU (D = C - E)",
            "Ventilator Usage (F)",
            "percentage ICU usage",
        ],
        inplace=True,
    )
    return case_df


def load_regional_cases(state):
    df = pd.read_excel(COVID_REGIONAL_URL)
    df.to_csv(COVID_REGIONAL_CSV)
    df = df[df.state == state]
    df.date = pd.to_datetime(
        df["date"], errors="coerce", format="%Y-%m-%d", infer_datetime_format=False
    )
    df["date_index"] = (df.date - COVID_BASE_DATE).dt.days
    df.sort_values(by=["date"], inplace=True)
    return df


if __name__ == "__main__":
    main()
