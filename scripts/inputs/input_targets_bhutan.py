"""
Script for loading Bhutan and Thimphu data into calibration targets and default.yml
"""

import pandas as pd
from sqlalchemy import DATE
from pathlib import Path

from autumn.settings import PROJECTS_PATH
from autumn.settings import INPUT_DATA_PATH
from autumn.core.utils.utils import update_timeseries
from autumn.models.covid_19.constants import COVID_BASE_DATETIME
from autumn.core.utils.utils import create_date_index
from autumn.model_features.curve.scale_up import scale_up_function

PROJECTS_PATH = Path(PROJECTS_PATH)
INPUT_DATA_PATH = Path(INPUT_DATA_PATH)

SM_SIR_BTN_TS = PROJECTS_PATH / "sm_sir" / "bhutan" / "bhutan" / "timeseries.json"
SM_SIR_THM_TS = PROJECTS_PATH / "sm_sir" / "bhutan" / "thimphu" / "timeseries.json"


DATA_PATH = INPUT_DATA_PATH / "covid_btn"
DATA_FILE = DATA_PATH / "Confirmed Cases for WHO reporting (68).xlsx"
WHO_DATA = INPUT_DATA_PATH / "covid_btn" / "data.csv"

TARGET_MAP_BTN = {
    "notifications": "detailed_cases",
    "infection_deaths": "detailed_cases_deaths",
    "hospital_occupancy": "detailed_cases_hospitalised",
}

TARGET_MAP_THM = {
    "notifications": "Date of diagnosis ",
}


def get_who_data(WHO_DATA):
    df = pd.read_csv(WHO_DATA)
    df = create_date_index(COVID_BASE_DATETIME, df, "ISO_START_DATE")
    df.columns

    cols = [
        "date",
        "date_index",
        "detailed_cases_deaths",
        "detailed_cases_hospitalised",
    ]

    df = df[cols]
    df = df.dropna(subset=cols[2:], how="all")

    all_days = pd.DataFrame({"date_index": list(range(1, 860))})
    all_days = pd.merge(all_days, df, how="left", on="date_index")

    df = all_days.ffill(limit=6)
    df = df.apply(lambda x: round(x / 7, 2) if x.name in cols[2:] else x)
    return df


def main():

    df = pd.read_excel(
        DATA_FILE, sheet_name="data sheet", usecols=["Date of diagnosis ", "District"]
    )
    df["date_index"] = df["Date of diagnosis "].apply(
        lambda d: (d - COVID_BASE_DATETIME).days
    )

    bhutan_cases = df.groupby("date_index", as_index=False).count()
    bhutan_cases = bhutan_cases[["date_index", "Date of diagnosis "]].rename(
        columns={"Date of diagnosis ": "detailed_cases"}
    )
    who_df = get_who_data(WHO_DATA)
    bhutan_cases = pd.merge(who_df, bhutan_cases, how="left", on="date_index")
    bhutan_cases = bhutan_cases.dropna(subset=bhutan_cases.columns[2:], how="all")

    thimphu_district = df["District"].apply(lambda s: "thimphu" in str(s).lower())
    thimphu_cases = df[thimphu_district].groupby("date_index", as_index=False).count()

    update_timeseries(TARGET_MAP_BTN, bhutan_cases, SM_SIR_BTN_TS)
    update_timeseries(TARGET_MAP_THM, thimphu_cases, SM_SIR_THM_TS)


if __name__ == "__main__":
    main()
