from pathlib import Path, PurePath

import pandas as pd
from autumn.core.utils.utils import create_date_index, update_timeseries
from autumn.models.covid_19.constants import COVID_BASE_DATETIME
from autumn.settings import INPUT_DATA_PATH, PROJECTS_PATH

URL = "https://govtstats.covid19nearme.com.au/data/all.csv"

COVID_AUS_DIRPATH = Path(INPUT_DATA_PATH, "covid_aus")


STATES = ["ACT", "VIC", "NSW", "WA", "SA", "TAS", "QLD", "NT"]

SM_SIR_PATH = {
    "NT": Path(PROJECTS_PATH, "sm_sir", "australia", "northern-territory", "timeseries.json")
}


TARGETS = {
    region: {
        "notifications": f"{region.lower()}_cases_local_last_24h",
        "hospital_occupancy": f"{region.lower()}_cases_hospital_not_icu",
        "icu_occupancy": f"{region.lower()}_cases_hospital_not_icu",
    }
    for region in STATES
}


def fetch_aus_data():

    return pd.read_csv(URL)


def preproces_mys_data(df):

    for region in STATES:

        df = create_date_index(COVID_BASE_DATETIME, df, "DATE")

        if region == "NT":
            update_timeseries(TARGETS[region], df, SM_SIR_PATH[region])


df = fetch_aus_data()
preproces_mys_data(df)
