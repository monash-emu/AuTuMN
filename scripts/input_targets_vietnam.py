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
COVID_VNM_TARGETS = os.path.join(PROJECTS_PATH, "covid_19", "vietnam", "vietnam", "timeseries.json")
COVID_VNM_HCMC_TARGETS = os.path.join(
    PROJECTS_PATH, "covid_19", "vietnam", "ho_chi_minh_city", "timeseries.json"
)
COVID_VMN_HCMC_CASES_CSV = os.path.join(INPUT_DATA_PATH, "covid_vnm", "cases.csv")
COVID_VNM_HCMC_TEST_CSV = os.path.join(INPUT_DATA_PATH, "covid_vnm", "testing.csv")
COVID_VNM_HCMC_URL = "https://docs.google.com/spreadsheets/d/11Z4HyBomVjI9VCbQrGFfti9CwAe_PZM6tbSzv7IA9LI/export?format=xlsx&id=11Z4HyBomVjI9VCbQrGFfti9CwAe_PZM6tbSzv7IA9LI"

TARGET_MAP_VNM = {
    "notifications": "new_cases",
    "infection_deaths": "new_deaths",
}

TARGET_MAP_VNM_HCMC = {
    "notifications": "notifications",
    "infection_deaths": "infection_deaths",
    "hospital_occupancy": "hospital_occupancy",
    "icu_occupancy":"icu_occupancy",
}


def preprocess_vnm_data():
    df = pd.read_csv(COVID_OWID)
    df = df[df.iso_code == "VNM"]
    df = create_date_index(COVID_BASE_DATETIME, df, "date")
    df = df[df.date <= pd.to_datetime("today").date()]

    return df

# Update VNM targets per OWID
df = preprocess_vnm_data()
update_timeseries(TARGET_MAP_VNM, df, COVID_VNM_TARGETS)

# Update HCMC targets
df_cases = pd.read_excel(COVID_VNM_HCMC_URL, usecols=[1,2,3,4,5])
df_cases = create_date_index(COVID_BASE_DATETIME, df_cases, "Unnamed:_1")
df_cases.to_csv(COVID_VMN_HCMC_CASES_CSV)
update_timeseries(TARGET_MAP_VNM_HCMC, df_cases, COVID_VNM_HCMC_TARGETS)
