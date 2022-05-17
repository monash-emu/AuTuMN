import pandas as pd

from pathlib import Path
from autumn.settings import PROJECTS_PATH
from autumn.settings import INPUT_DATA_PATH
from autumn.tools.utils.utils import update_timeseries
from autumn.models.covid_19.constants import COVID_BASE_DATETIME
from autumn.tools.utils.utils import create_date_index


GITHUB_MOH = "https://raw.githubusercontent.com/MoH-Malaysia/covid19-public/main/epidemic/"


COVID_MYS_DIRPATH = Path(INPUT_DATA_PATH, "covid_mys")


REGIONS = ["malaysia", "sabah", "selangor", "johor", "kuala_lumpur", "penang"]
REGION_PATH = {
    region: Path(PROJECTS_PATH, "covid_19", "malaysia", region, "timeseries.json")
    for region in REGIONS
}

SM_SIR_PATH = {
    "malaysia": Path(PROJECTS_PATH, "sm_sir", "malaysia", "malaysia", "timeseries.json")
}

FILES = ["cases_malaysia", "deaths_malaysia", "hospital", "icu", "cases_state", "deaths_state"]

TARGETS = {
    region: {
        "notifications": "cases_new",
        "infection_deaths": "deaths_new",
        "hospital_occupancy": "hosp_covid",
        "icu_occupancy": "icu_covid",
    }
    for region in REGIONS
}


def fetch_mys_data():

    for file in FILES:
        df = pd.read_csv(GITHUB_MOH + file + ".csv")
        df.to_csv(COVID_MYS_DIRPATH / (file +".csv"), index=False)

    return


def preproces_mys_data():

    for region in REGIONS:

        for file in FILES:

            file_path = COVID_MYS_DIRPATH /  (file + ".csv")
            df = pd.read_csv(file_path)
            df = create_date_index(COVID_BASE_DATETIME, df, "date")
            df.to_csv(file_path, index=False)

            if region == "malaysia":

                if file in {"hospital", "icu"}:
                    df = df.groupby("date_index").sum()
                    df.reset_index(inplace=True)
                if file in {"cases_malaysia", "deaths_malaysia", "hospital", "icu"}:
                    df = df[df.date_index >= 50]
                    update_timeseries(TARGETS[region], df, SM_SIR_PATH[region])

fetch_mys_data()
preproces_mys_data()
