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

from autumn import settings
from autumn.settings import Region

# From WHO google drive folder
CASE_DATA_URL = "https://docs.google.com/spreadsheets/d/1FbYG8szgbvw3pjULWEHDZ8uhu5BooV_-cygLn7iLaxA/export?format=xlsx&id=1FbYG8szgbvw3pjULWEHDZ8uhu5BooV_-cygLn7iLaxA"  # shareable link
COVID_MYS_DEATH_URL = "https://docs.google.com/spreadsheets/d/1iv0veITNSKxpoVvY2WDOePgPC0LvFmOSjsYEq-5BsMM/export?format=xlsx&id=1iv0veITNSKxpoVvY2WDOePgPC0LvFmOSjsYEq-5BsMM"
COVID_REGIONAL_URL = "https://docs.google.com/spreadsheets/d/1ouWp2ge5zrVh1gCDPONN-TMaf1b7TqRr4f07UZLWSdg/export?format=xlsx&id=1ouWp2ge5zrVh1gCDPONN-TMaf1b7TqRr4f07UZLWSdg"

COVID_MYS_DIRPATH = os.path.join(settings.folders.INPUT_DATA_PATH, "covid_mys")
COVID_MYS_CASE_CSV = os.path.join(COVID_MYS_DIRPATH, "COVID_MYS_CASE.csv")
COVID_MYS_DEATH_CSV = os.path.join(COVID_MYS_DIRPATH, "COVID_MYS_DEATH.csv")
COVID_REGIONAL_CSV = os.path.join(COVID_MYS_DIRPATH, "COVID_REGIONAL.csv")


COVID_BASE_DATE = pd.datetime(2019, 12, 31)
REGION = ["malaysia", "sabah", "selangor", "johor", "kuala_lumpur", "penang"]
REGION_PATH = {
    region: os.path.join(settings.folders.PROJECTS_PATH, "covid_19", "malaysia", region)
    for region in REGION
}

TARGETS = {
    region: {"notifications": region + "_case", "infection_deaths": region + "_death"}
    for region in REGION
}
TARGETS["malaysia"]["icu_occupancy"] = "malaysia_ICU"


def main():

    death_df = get_death()
    regional_df = load_regional_cases()
    mys_case_df = load_mys()

    df = pd.merge(
        mys_case_df, regional_df, how="outer", left_on=["date_index"], right_on=["date_index"]
    )
    df = pd.merge(df, death_df, how="outer", left_on=["date_index"], right_on=["date_index"])

    for region in REGION:

        file_path = os.path.join(REGION_PATH[region], "timeseries.json")
        with open(file_path, mode="r") as f:
            targets = json.load(f)
        for key, val in TARGETS[region].items():
            # Drop the NaN value rows from df before writing data.
            temp_df = df[["date_index", val]].dropna(0, subset=[val])

            targets[key]["times"] = list(temp_df["date_index"])
            targets[key]["values"] = list(temp_df[val])
        with open(file_path, "w") as f:
            json.dump(targets, f, indent=2)


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
    death_df.state = death_df.state.str.lower()

    def fix_regex(each_row, state):

        x = re.findall(state + r"=(\d+)", each_row)
        if len(x) > 0:
            return int(x[0])
        else:
            return 0

    death_df["sabah_death"] = [fix_regex(each, "sabah") for each in death_df.state]
    death_df["selangor_death"] = [fix_regex(each, "selangor") for each in death_df.state]
    death_df["johor_death"] = [fix_regex(each, "johor") for each in death_df.state]
    death_df["penang_death"] = [fix_regex(each, "pulau pinang") for each in death_df.state]
    death_df["kuala_lumpur_death"] = [fix_regex(each, "kl") for each in death_df.state]

    # If it's just one state then copy over the deaths
    death_df.loc[death_df.state == "sabah", "sabah_death"] = death_df["Death per day"]
    death_df.loc[death_df.state == "selangor", "selangor_death"] = death_df["Death per day"]
    death_df.loc[death_df.state == "johor", "johor_death"] = death_df["Death per day"]
    death_df.loc[death_df.state == "pulau pinang", "penang_death"] = death_df["Death per day"]
    death_df.loc[death_df.state == "kl", "kuala_lumpur_death"] = death_df["Death per day"]

    death_df.rename(columns={"Death per day": "malaysia_death"}, inplace=True)

    return death_df[
        [
            "date_index",
            "malaysia_death",
            "sabah_death",
            "selangor_death",
            "johor_death",
            "penang_death",
            "kuala_lumpur_death",
        ]
    ]


def load_regional_cases():
    df = pd.read_excel(COVID_REGIONAL_URL)
    df.to_csv(COVID_REGIONAL_CSV)
    df["state"] = df.state.str.lower()
    df["state"].replace({"kuala lumpur": "kuala_lumpur", "pulau pinang": "penang"}, inplace=True)
    df.date = pd.to_datetime(
        df["date"], errors="coerce", format="%Y-%m-%d", infer_datetime_format=False
    )
    df["date_index"] = (df.date - COVID_BASE_DATE).dt.days
    df.sort_values(by=["date"], inplace=True)
    df = df.pivot(index="date_index", columns="state", values="local_cases")[REGION[1:]]
    df.rename(
        columns={
            "sabah": "sabah_case",
            "selangor": "selangor_case",
            "johor": "johor_case",
            "kuala_lumpur": "kuala_lumpur_case",
            "penang": "penang_case",
        },
        inplace=True,
    )
    return df


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
            "New Cases (A)": "malaysia_case",
            "Imported cases (B)": "IC",
            "Total ICU Usage including ventilator usage (E)": "malaysia_ICU",
        },
        inplace=True,
    )

    return case_df[["date_index", "malaysia_case", "malaysia_ICU"]]


if __name__ == "__main__":
    main()
