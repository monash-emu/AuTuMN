"""
Script for loading Bhutan and Thimphu data into calibration targets and default.yml
"""
import os
from typing import List
import pandas as pd
from sqlalchemy import DATE

from autumn.settings import PROJECTS_PATH
from autumn.settings import INPUT_DATA_PATH
from autumn.tools.utils.utils import update_timeseries
from autumn.models.covid_19.constants import COVID_BASE_DATETIME
from autumn.tools.utils.utils import create_date_index


SM_SIR_BTN_TS = os.path.join(
    PROJECTS_PATH, "sm_sir", "bhutan", "bhutan", "timeseries.json"
)
SM_SIR_THM_TS = os.path.join(
    PROJECTS_PATH, "sm_sir", "bhutan", "thimphu", "timeseries.json"
)


DATA_PATH = os.path.join(INPUT_DATA_PATH, "covid_btn")
DATA_FILE = os.path.join(DATA_PATH, "Confirmed Cases for WHO reporting (68).xlsx")

TARGET_MAP_BTN = {
    "notifications": "Date of diagnosis ",
}

TARGET_MAP_THM = {
    "notifications": "Date of diagnosis ",
}


def main():

    df = pd.read_excel(
        DATA_FILE, sheet_name="data sheet", usecols=["Date of diagnosis ", "District"]
    )
    df["date_index"] = df["Date of diagnosis "].apply(
        lambda d: (d - COVID_BASE_DATETIME).days
    )

    bhutan_cases = df.groupby("date_index", as_index=False).count()

    thimphu_district = df["District"].apply(lambda s: "thimphu" in str(s).lower())
    thimphu_cases = df[thimphu_district].groupby("date_index", as_index=False).count()

    update_timeseries(TARGET_MAP_BTN, bhutan_cases, SM_SIR_BTN_TS)
    update_timeseries(TARGET_MAP_THM, thimphu_cases, SM_SIR_THM_TS)


if __name__ == "__main__":
    main()
