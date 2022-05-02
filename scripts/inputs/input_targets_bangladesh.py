"""
Script for loading Bangladesh, Dhaka and Cox's Bazar data into calibration targets and default.yml
NOTE you will need to pip instal lxml to run this script

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

DATA_FILE = os.path.join(DATA_PATH, "Bangladesh COVID-19 template.xlsx")

TARGET_MAP_BGD = {
    "notifications": "bgd_confirmed_cases",
    "infection_deaths": "bgd_death",
    "hospital_admissions": "bgd_hospital_admissions",
}

TARGET_MAP_DHK = {
    "notifications": "dhk_confirmed_cases",
    "infection_deaths": "dhk_death",
    "hospital_admissions": "dhk_hospital_admissions",
}


TARGET_MAP_COXS = {
    "notifications": "confirmed_cases",
    "infection_deaths": "death",
    "icu_occupancy": "admitted_cases_at_icu/hdu_in_district",
    "hospital_occupancy": "admitted_cases_at_itc",
}


def main():

    save_to_excel(DHK_DATA + BGD_DATA)

    df = pd.read_excel(DATA_FILE)
    df = create_date_index(COVID_BASE_DATETIME, df, "Date")

    update_timeseries(TARGET_MAP_BGD, df, SM_SIR_BGD_TS)
    update_timeseries(TARGET_MAP_DHK, df, SM_SIR_DHK_TS)

    # Cox's bazar
    df = pd.read_excel(COXS_DATA, skipfooter=1, usecols=[1, 2, 3, 4, 5, 6])
    df.loc[(~df["Confirmed cases"].isna()) & (df["Death"].isna()), "Death"] = 0
    df = create_date_index(COVID_BASE_DATETIME, df, "Unnamed: 1")
    update_timeseries(TARGET_MAP_COXS, df, SM_SIR_COXS_TS)


def save_to_excel(file_paths: List) -> None:
    """Convert dashboard files to csv"""

    for file in file_paths:
        try:
            pd.read_html(file)[0].to_csv(file, index=False)
        except:
            assert len(pd.read_csv(file)) > 0, f"Download {file} again"

    return None


if __name__ == "__main__":
    main()
