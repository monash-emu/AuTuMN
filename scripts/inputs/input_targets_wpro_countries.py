import pandas as pd
import json

from pathlib import Path, PurePath
from autumn.settings import PROJECTS_PATH, BASE_PATH, INPUT_DATA_PATH
from autumn.core.utils.utils import update_timeseries
from autumn.models.covid_19.constants import COVID_BASE_DATETIME
from autumn.core.utils.utils import create_date_index


WPRO_URL = (
    "https://who.maps.arcgis.com/sharing/rest/content/items/f2e991c1fce54932b1406a3efbde5f27/data"
)

WPRO_DIR_PATH = Path(INPUT_DATA_PATH, "covid_wpro")
WPRO_DATA_CSV = WPRO_DIR_PATH / "wpro_data.csv"


WPRO_LIST = json.load(open(BASE_PATH / "autumn" / "wpro_list.json"))
regions = WPRO_LIST["region"]
WPRO_REGION_PATH = [BASE_PATH / x for x in ["/".join(x.split(".")[:-1]) for x in WPRO_LIST["path"]]]
WPRO_REGION_PATH = dict(zip(regions, WPRO_REGION_PATH))


WPRO_REGIONS = {
    "malaysia": "MYS",
    "australia": "AUS",
    "philippines": "PHL",
    "vietnam": "VNM",
    "japan": "JPN",
    "mongolia": "MNG",
    "new-zealand": "NZL",
    "south-korea": "KOR",
    "singapore": "SGP",
    "china": "CHN",
}

TARGETS = {
    "notifications": "newcases",
    "infection_deaths": "newdeaths",
    "cumulative_infection_deaths": "totaldeaths",
    "cumulative_incidence": "totalcases",
    # "hospital_occupancy": "hosp_covid",
    # "icu_occupancy": "icu_covid",
}


def fetch_wpro_data(str: WPRO_URL) -> None:

    pd.read_csv(WPRO_URL).to_csv(WPRO_DATA_CSV, index=False)
    return None


def preproces_wpro_data():

    df = pd.read_csv(WPRO_DATA_CSV)
    df = create_date_index(COVID_BASE_DATETIME, df, "DateReport")

    for region in WPRO_REGIONS:

        df_region = df[df["iso_3_code"] == WPRO_REGIONS[region]]
        timeseries_file = WPRO_REGION_PATH[region] / "timeseries.json"
        update_timeseries(TARGETS, df_region, timeseries_file)


fetch_wpro_data(WPRO_URL)
preproces_wpro_data()
