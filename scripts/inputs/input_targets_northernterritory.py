from pathlib import Path

import pandas as pd
from autumn.core.utils.utils import create_date_index, update_timeseries
from autumn.models.covid_19.constants import COVID_BASE_DATETIME
from autumn.settings import COVID_BASE_DATETIME, INPUT_DATA_PATH, PROJECTS_PATH

COVID_AUS_DIRPATH = Path(INPUT_DATA_PATH, "covid_au")
URL = "https://govtstats.covid19nearme.com.au/data/all.csv"
NT_DATA = COVID_AUS_DIRPATH / "NT data.xlsx"

LOCAL_RUN = False
STATES = ["ACT", "VIC", "NSW", "WA", "SA", "TAS", "QLD", "NT"]

SM_SIR_PATH = {
    "NT": Path(PROJECTS_PATH, "sm_sir", "australia", "northern_territory", "timeseries.json")
}


TARGETS_ABC = {
    region: {
        "notifications_abc": f"{region.lower()}_cases_local_last_24h",
        "hospital_occupancy_abc": f"{region.lower()}_cases_hospital_not_icu",
        "icu_occupancy_abc": f"{region.lower()}_cases_hospital_not_icu",
    }
    for region in STATES
}

TARGETS_NT = {
    "notifications": "total_cases",
    "hospital_admission": "total",
    "icu_admission": "freq.",
}


nt_param_map = {
    "Cases": {
        "sheet_name": "Cases Time Series",
        "skipfooter": 2,
        "usecols": [0, 3],
        "datecol": "notdate",
    },
    "Hosp_adm": {
        "sheet_name": "Hosp admit time series",
        "skipfooter": 2,
        "skiprows": [0, 1],
        "usecols": [1, 4],
        "datecol": "hosp_admit_date",
    },
    "ICU_adm": {
        "sheet_name": "Hosp admit ICU time series",
        "skipfooter": 2,
        "skiprows": [0, 1, 3],
        "usecols": [1, 2],
        "datecol": "icu_date",
    },
}


def main():
    df = fetch_abc_aus_data()
    preproces_abc_nt_data(df)

    for each in nt_param_map:
        *kwargs, date = nt_param_map[each].items()
        df = pd.read_excel(NT_DATA, **dict(kwargs))
        df = create_date_index(COVID_BASE_DATETIME, df, date[1])

        if LOCAL_RUN or each == "Cases":
            update_timeseries(TARGETS_NT, df, SM_SIR_PATH["NT"])


def fetch_abc_aus_data():

    return pd.read_csv(URL)


def preproces_abc_nt_data(df):

    for region in STATES:

        df = create_date_index(COVID_BASE_DATETIME, df, "DATE")

        if region == "NT":
            update_timeseries(TARGETS_ABC[region], df, SM_SIR_PATH[region])


if __name__ == "__main__":
    main()
