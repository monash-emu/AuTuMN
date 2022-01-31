"""
Script for loading NPL data into calibration targets and default.yml

"""

import os
import pandas as pd

from autumn.settings import PROJECTS_PATH
from autumn.settings import INPUT_DATA_PATH
from autumn.tools.utils.utils import update_timeseries
from autumn.tools.utils.utils import COVID_BASE_DATETIME
from autumn.tools.utils.utils import create_date_index


# Use OWID csv for notification and death numbers.
COVID_OWID = os.path.join(INPUT_DATA_PATH, "owid", "owid-covid-data.csv")
COVID_VNM_TS = os.path.join(PROJECTS_PATH, "covid_19", "vietnam", "vietnam", "timeseries.json")
COVID_HCMC_TS = os.path.join(
    PROJECTS_PATH, "covid_19", "vietnam", "ho_chi_minh_city", "timeseries.json"
)
COVID_HAN_TS = os.path.join(PROJECTS_PATH, "covid_19", "vietnam", "hanoi", "timeseries.json")

SM_SIR_HCMC_TS = os.path.join(
    PROJECTS_PATH, "sm_sir", "vietnam", "ho_chi_minh_city", "timeseries.json"
)


HCMC_DATA_CSV = os.path.join(INPUT_DATA_PATH, "covid_vnm", "cases.csv")
HANOI_DATA_CSV = os.path.join(INPUT_DATA_PATH, "covid_vnm", "hanoi_data.csv")
HCMC_DATA_URL = "https://docs.google.com/spreadsheets/d/11Z4HyBomVjI9VCbQrGFfti9CwAe_PZM6tbSzv7IA9LI/export?format=xlsx&id=11Z4HyBomVjI9VCbQrGFfti9CwAe_PZM6tbSzv7IA9LI"
HANOI_DATA_URL = "https://docs.google.com/spreadsheets/d/10QEfHx9AeV6V4AoFaxHnNiRpAyavUYboKwwwqcKIsTo/export?format=xlsx&id=10QEfHx9AeV6V4AoFaxHnNiRpAyavUYboKwwwqcKIsTo"


TARGET_MAP_VNM = {
    "notifications": "new_cases",
    "infection_deaths": "new_deaths",
}

TARGET_MAP_HCMC = {
    "notifications": "notifications",
    "infection_deaths": "infection_deaths",
    "hospital_occupancy": "hospital_occupancy",
    "icu_occupancy": "icu_occupancy",
}

TARGET_MAP_HANOI = {
    "notifications": "notifications",
    "icu_occupancy": "icu_occupancy",
    "infection_deaths": "infection_deaths",
}


def preprocess_vnm_data():
    df = pd.read_csv(COVID_OWID)
    df = df[df.iso_code == "VNM"]
    df = create_date_index(COVID_BASE_DATETIME, df, "date")
    df = df[df["new_deaths"] >= 0]
    df = df[df.date <= pd.to_datetime("today").date()]

    return df


# Update VNM targets per OWID
df = preprocess_vnm_data()
update_timeseries(TARGET_MAP_VNM, df, COVID_VNM_TS)

# Update HCMC targets
df_cases = pd.read_excel(HCMC_DATA_URL, usecols=[1, 2, 3, 4, 5])
df_cases = create_date_index(COVID_BASE_DATETIME, df_cases, "Unnamed:_1")
df_cases.to_csv(HCMC_DATA_CSV)
update_timeseries(TARGET_MAP_HCMC, df_cases, COVID_HCMC_TS)
update_timeseries(TARGET_MAP_HCMC, df_cases, SM_SIR_HCMC_TS)

# Update HANOI targets
df_cases = pd.read_excel(HANOI_DATA_URL, usecols=[0, 1, 2, 3])
df_cases = create_date_index(COVID_BASE_DATETIME, df_cases, "date")
df_cases.to_csv(HANOI_DATA_CSV)
update_timeseries(TARGET_MAP_HANOI, df_cases, COVID_HAN_TS)
