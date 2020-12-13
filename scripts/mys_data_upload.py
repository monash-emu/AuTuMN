##!/usr/bin/env python
"""
Script for loading MYS data into calibration targets and default.yml

Manually download file "Total Death Covid19 in Malaysia_updated ..."
from https://drive.google.com/drive/u/1/folders/1C-OsJbjdhQmpmedBZ8XuM65ANBO22dXc
to ./data/inputs/covid_mys/COVID_MYS_DEATH.csv for now!
"""
import os
import sys
import json
import yaml
from datetime import datetime
from google_drive_downloader import GoogleDriveDownloader as gdd
from autumn import constants
import pandas as pd
import regex as re


# From WHO google drive folder
CASE_DATA_URL = "1mnZcmj2jfmrap1ytyg_ErD1DZ7zDptyJ"  # shareable link

COVID_MYS_DIRPATH = os.path.join(constants.INPUT_DATA_PATH, "covid_mys")
COVID_MYS_CASE_XLSX = os.path.join(COVID_MYS_DIRPATH, "COVID_MYS_CASE.xlsx")
COVID_MYS_DEATH_CSV = os.path.join(COVID_MYS_DIRPATH, "COVID_MYS_DEATH.csv")
COVID_SABAH_CSV = os.path.join(COVID_MYS_DIRPATH, "COVID_REGIONAL.csv")


gdd.download_file_from_google_drive(
    file_id=CASE_DATA_URL, dest_path=COVID_MYS_CASE_XLSX, overwrite=True
)

COVID_BASE_DATE = pd.datetime(2019, 12, 31)
REGION_MYS = os.path.join(constants.APPS_PATH, "covid_19", "regions", "malaysia")
REGION_SABAH = os.path.join(constants.APPS_PATH, "covid_19", "regions", "sabah")


TARGETS_MYS = {
    "notifications": "NC",
    "icu_occupancy": "ICU",
    "infection_deaths": "Death per day",
}


TARGETS_SABAH = {"notifications": "local_cases", "infection_deaths": "sabah_death"}


def main():

    update_mys_calibration()
    update_sabah_calibration()


def update_mys_calibration():
    """
    Update Malaysia's calibration targets
    """

    df = load_mys_data()

    file_path = os.path.join(REGION_MYS, "targets.json")
    with open(file_path, mode="r") as f:
        targets = json.load(f)
    for key, val in TARGETS_MYS.items():
        # Drop the NaN value rows from df before writing data.
        temp_df = df[["date_index", val]].dropna(0, subset=[val])

        targets[key]["times"] = list(temp_df["date_index"])
        targets[key]["values"] = list(temp_df[val])
    with open(file_path, "w") as f:
        json.dump(targets, f, indent=2)
    with open(REGION_MYS + "\\params\\default.yml", "r") as f:
        default = f.readlines()
    end = default.index("  case_timeseries:\n")
    default = default[:end]

    with open(REGION_MYS + "\\params\\default.yml", "r") as f:
        tmp = yaml.load(f, Loader=yaml.FullLoader)
    tmp = {key: val for key, val in tmp.items() if key == "importation"}

    temp_df = df[["date_index", "IC"]].dropna(0, subset=["IC"])
    temp_df = temp_df[temp_df.date_index != 94]
    tmp["importation"]["case_timeseries"]["times"] = list(temp_df["date_index"])
    tmp["importation"]["case_timeseries"]["values"] = list(temp_df["IC"])

    with open(REGION_MYS + "\\params\\default.yml", "w") as f:
        yaml.dump(tmp, f)
    with open(REGION_MYS + "\\params\\default.yml", "r") as f:
        append = f.readlines()

    end = append.index("  movement_prop: null\n")
    default = default + append[1:end]

    with open(REGION_MYS + "\\params\\default.yml", "w") as f:
        f.writelines(default)


def load_mys_data():

    case_df = pd.read_excel(COVID_MYS_CASE_XLSX, engine="openpyxl")
    death_df = get_death()

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
            "percentage ICU usage",
        ],
        inplace=True,
    )

    df = pd.merge(case_df, death_df, how="left", left_on=["date_index"], right_on=["date_index"])

    return df


def get_death():
    death_df = pd.read_csv(COVID_MYS_DEATH_CSV)
    for column in death_df.columns:
        if column not in {"Date", "Death per day", "state"}:
            death_df.drop(columns=[column], inplace=True)
    death_df.Date = death_df.Date.str[:] + "-2020"
    death_df.Date = pd.to_datetime(
        death_df["Date"], errors="coerce", format="%d-%b-%Y", infer_datetime_format=False
    )
    death_df["date_index"] = (death_df.Date - COVID_BASE_DATE).dt.days
    death_df = death_df[death_df.Date.notnull() & death_df["Death per day"].notnull()]
    death_df.loc[death_df.state.isna(), "state"] = ""

    def fix_regex(each_row):

        x = re.findall(r"Sabah=(\d+),", each_row)
        if len(x) > 0:
            return int(x[0])
        else:
            return 0

    death_df["sabah_death"] = [fix_regex(each) for each in death_df.state]

    death_df.loc[death_df.state == "Sabah", "sabah_death"] = death_df["Death per day"]
    return death_df[["date_index", "Death per day", "state", "sabah_death"]]


def update_sabah_calibration():
    """
    Update Sabah's calibration targets
    """

    df = pd.read_csv(COVID_SABAH_CSV)
    df = df[df.state == "Sabah"]
    df.date = pd.to_datetime(
        df["date"], errors="coerce", format="%d/%m/%Y", infer_datetime_format=False
    )
    df["date_index"] = (df.date - COVID_BASE_DATE).dt.days
    df.sort_values(by=["date"], inplace=True)

    death_data = get_death()
    death_data = death_data[["sabah_death", "date_index"]]

    df = pd.merge(df, death_data, how="left", left_on=["date_index"], right_on=["date_index"])

    file_path = os.path.join(REGION_SABAH, "targets.json")
    with open(file_path, mode="r") as f:
        targets = json.load(f)
    for key, val in TARGETS_SABAH.items():
        # Drop the NaN value rows from df before writing data.
        temp_df = df[["date_index", val]].dropna(0, subset=[val])

        targets[key]["times"] = list(temp_df["date_index"])
        targets[key]["values"] = list(temp_df[val])
    with open(file_path, "w") as f:
        json.dump(targets, f, indent=2)

    with open(REGION_SABAH + "\\params\\default.yml", "r") as f:
        default = f.readlines()
    end = default.index("  case_timeseries:\n")
    default = default[:end]

    with open(REGION_SABAH + "\\params\\default.yml", "r") as f:
        tmp = yaml.load(f, Loader=yaml.FullLoader)
    tmp = {key: val for key, val in tmp.items() if key == "importation"}
    temp_df = df[["date_index", "import_cases"]].dropna(0, subset=["import_cases"])
    tmp["importation"]["case_timeseries"]["times"] = list(temp_df["date_index"])
    tmp["importation"]["case_timeseries"]["values"] = list(temp_df["import_cases"])

    with open(REGION_SABAH + "\\params\\default.yml", "w") as f:
        yaml.dump(tmp, f)
    with open(REGION_SABAH + "\\params\\default.yml", "r") as f:
        append = f.readlines()

    end = append.index("  movement_prop: null\n")
    default = default + append[1:end]

    with open(REGION_SABAH + "\\params\\default.yml", "w") as f:
        f.writelines(default)


if __name__ == "__main__":
    main()
