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
COVID_VNM_HCMC_URL = "https://docs.google.com/spreadsheets/d/1CKCT9uOfKNuF4KNs8vKnr3y-hokkg9kv/export?format=xlsx&id=1CKCT9uOfKNuF4KNs8vKnr3y-hokkg9kv"

TARGET_MAP_VNM = {
    "notifications": "new_cases",
    "infection_deaths": "new_deaths",
}

TARGET_MAP_VNM_HCMC = {
    "notifications": "daily_reported_cases",
    "infection_deaths": "daily_death",
    "hospital_occupancy": "total_severe_cases",
}


def preprocess_vnm_data():
    df = pd.read_csv(COVID_OWID)
    df = df[df.iso_code == "VNM"]
    df = create_date_index(COVID_BASE_DATETIME, df, "date")
    df = df[df.date <= pd.to_datetime("today")]

    return df


df = preprocess_vnm_data()
update_timeseries(TARGET_MAP_VNM, df, COVID_VNM_TARGETS)

df_cases = pd.read_excel(COVID_VNM_HCMC_URL, sheet_name=["Reported cases"])["Reported cases"]
df_cases = create_date_index(COVID_BASE_DATETIME, df_cases, "Date")
df_cases.to_csv(COVID_VMN_HCMC_CASES_CSV)
update_timeseries(TARGET_MAP_VNM_HCMC, df_cases, COVID_VNM_HCMC_TARGETS)

df_testing = pd.read_excel(COVID_VNM_HCMC_URL, skiprows=[0, 2], sheet_name=["Testing"])["Testing"]
df_testing = create_date_index(COVID_BASE_DATETIME, df_testing, "Date")
df_testing.rename(columns={"sum.1": "daily_test"}, inplace=True)
df_testing["region"] = "Ho Chi Minh City"
df_testing = df_testing[["date", "date_index", "region", "daily_test"]][1:]
df_testing.to_csv(COVID_VNM_HCMC_TEST_CSV)
