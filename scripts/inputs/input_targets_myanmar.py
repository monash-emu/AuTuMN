"""
Script for loading Myanmar data into calibration targets and default.yml
"""

import json
import os
import pandas as pd
from datetime import datetime

from autumn.settings import PROJECTS_PATH

from autumn.settings import INPUT_DATA_PATH

from autumn.tools.utils.utils import update_timeseries, create_date_index

# start date to calculate time since Dec 31, 2019
COVID_BASE_DATETIME = datetime(2019, 12, 31, 0, 0, 0)


COVID_MMR_TARGETS = os.path.join(PROJECTS_PATH, "covid_19", "myanmar", "timeseries.json")
SM_SIR_PATH = os.path.join(PROJECTS_PATH, "sm_sir", "myanmar", "myanmar", "timeseries.json")
COVID_MMR_DATA = os.path.join(INPUT_DATA_PATH, "covid_mmr", "cases.csv")

URL = "https://docs.google.com/spreadsheets/d/1VeUof9_-s0bsndo8tLsCwnAhkUUZgsdV-r980gumMPA/export?format=csv&id=1VeUof9_-s0bsndo8tLsCwnAhkUUZgsdV-r980gumMPA"

TARGETS = {
    "notifications": "cases",
    "infection_deaths": "deaths",
}

mmr_df = pd.read_csv(URL)
mmr_df.to_csv(COVID_MMR_DATA)

str_col = ["Tests", "Cases", "Recovered", "Negative"]

mmr_df[str_col] = mmr_df[str_col].replace(to_replace=r",", value="", regex=True)
mmr_df[str_col] = mmr_df[str_col].apply(pd.to_numeric)
mmr_df["Date"] = pd.to_datetime(mmr_df["Date"])
mmr_df = create_date_index(COVID_BASE_DATETIME, mmr_df, "Date")

update_timeseries(TARGETS, mmr_df, COVID_MMR_TARGETS)
update_timeseries(TARGETS, mmr_df, SM_SIR_PATH)
