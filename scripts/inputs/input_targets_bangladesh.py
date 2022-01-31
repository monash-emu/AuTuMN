"""
Script for loading Bangladesh, Dhaka and Cox's Bazar data into calibration targets and default.yml
NOTE you will need to pip instal lxml to run this script

"""
import os
from typing import List
import pandas as pd

from autumn.settings import PROJECTS_PATH
from autumn.settings import INPUT_DATA_PATH
from autumn.tools.utils.utils import update_timeseries
from autumn.tools.utils.utils import COVID_BASE_DATETIME
from autumn.tools.utils.utils import create_date_index


SM_SIR_BGD_TS = os.path.join(PROJECTS_PATH, "sm_sir", "bangladesh", "bangladesh", "timeseries.json")
SM_SIR_DHK_TS = os.path.join(PROJECTS_PATH, "sm_sir", "bangladesh", "dhaka", "timeseries.json")
SM_SIR_COXS_TS = os.path.join(
    PROJECTS_PATH, "sm_sir", "bangladesh", "coxs_bazar", "timeseries.json"
)

DATA_PATH = os.path.join(INPUT_DATA_PATH, "covid_bgd")

FILES = os.listdir(DATA_PATH)

BGD_DATA = [os.path.join(DATA_PATH, file) for file in FILES if "BGD" in file]
DHK_DATA = [os.path.join(DATA_PATH, file) for file in FILES if "DHK" in file]
COXS_DATA = os.path.join(DATA_PATH, "COVID-19 Data for modelling.xlsx")


TARGET_MAP_BGD = {
    "notifications": "confirmed_case",
    "infection_deaths": "death",
}

TARGET_MAP_DHK = {
    "notifications": "confirmed_case",
    "infection_deaths": "death",
}


TARGET_MAP_COXS = {
    "notifications": "confirmed_cases",
    "infection_deaths": "death",
    "icu_admissions": "admitted_cases_at_icu/hdu_in_district",
}


def main():

    save_to_excel(DHK_DATA + BGD_DATA)

    # Bangladesh & Dhaka
    for target in {"deaths", "cases"}:

        update_region(SM_SIR_BGD_TS, BGD_DATA, TARGET_MAP_BGD, target)
        update_region(SM_SIR_DHK_TS, DHK_DATA, TARGET_MAP_DHK, target)

    # Cox's bazar
    df = pd.read_excel(COXS_DATA, skipfooter=1, usecols=[1, 2, 3, 4, 5, 6])
    create_date_index(COVID_BASE_DATETIME, df, "Unnamed: 1")
    update_timeseries(TARGET_MAP_COXS, df, SM_SIR_COXS_TS)


def save_to_excel(file_paths: List) -> None:
    """Convert dashboard files to csv"""

    for file in file_paths:
        try:
            pd.read_html(file)[0].to_csv(file, index=False)
        except:
            assert len(pd.read_csv(file)) > 0, f"Download {file} again"

    return None


def get_data_file(dtype: str, datafile: List[str]) -> list:
    return [file for file in datafile if dtype.lower() in file.lower()]


def create_dates(df: pd.DataFrame) -> pd.DataFrame:

    # Find the length and end date to create a datetime column
    periods = len(df)
    end_date = pd.to_datetime(f'{df.tail(1)["Category"].values[0]}-2022')
    df["date"] = pd.date_range(end=end_date, periods=periods).tolist()

    # Compare and remove any non matching rows
    df["compare"] = (
        df["date"].dt.month_name().astype(str)
        + "-"
        + df["date"].dt.day.astype(str).str.rjust(2, "0")
    )
    filter_dates = df["Category"] == df["compare"]
    df = df[filter_dates]

    return df


def update_region(timeseries, region_data, target_map, target) -> None:

    df = pd.read_csv(get_data_file(target, region_data)[0])
    df = create_dates(df)

    df = create_date_index(COVID_BASE_DATETIME, df, "date")
    update_timeseries(target_map, df, timeseries)
    return None


if __name__ == "__main__":
    main()
