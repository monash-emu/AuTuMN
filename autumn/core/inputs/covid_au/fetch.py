"""
This file imports Google mobility data and saves it to disk as a CSV.
"""

import pandas as pd

from pathlib import Path
from autumn.settings import INPUT_DATA_PATH

INPUT_DATA_PATH = Path(INPUT_DATA_PATH)

# From covid19data.com.au GitHub https://github.com/M3IT/COVID-19_Data
DATA_URL = "https://raw.githubusercontent.com/M3IT/COVID-19_Data/master/Data/COVID_AU_state_daily_change.csv"
COVID_AU_DIRPATH = INPUT_DATA_PATH / "covid_au"
COVID_AU_CSV_PATH = COVID_AU_DIRPATH / "COVID_AU_state_daily_change.csv"

YOUGOV_URL = (
    "https://github.com/YouGov-Data/covid-19-tracker/raw/master/data/australia.zip"
)
COVID_AU_YOUGOV = COVID_AU_DIRPATH / "yougov_australia.csv"

COVID_LGA_CSV_PATH = COVID_AU_DIRPATH / "lga_test.csv"
MOBILITY_DIRPATH = INPUT_DATA_PATH / "mobility"
MOBILITY_LGA_PATH = (
    MOBILITY_DIRPATH / "LGA to Cluster mapping dictionary with proportions.csv"
)

COVID_VAC_COV_CSV = COVID_AU_DIRPATH / "vac_cov.csv"
COVID_VIDA_VAC_CSV = COVID_AU_DIRPATH / "vida_vac.secret.csv"
COVID_VIDA_POP_CSV = COVID_AU_DIRPATH / "vida_pop.csv"


CLUSTER_MAP = {
    1: "NORTH_METRO",
    2: "SOUTH_EAST_METRO",
    3: "SOUTH_METRO",
    4: "WEST_METRO",
    5: "BARWON_SOUTH_WEST",
    6: "GIPPSLAND",
    7: "GRAMPIANS",
    8: "HUME",
    9: "LODDON_MALLEE",
    0: "VIC",
}


def fetch_covid_au_data():
    pd.read_csv(DATA_URL).to_csv(COVID_AU_CSV_PATH)

    pd.read_csv(YOUGOV_URL).to_csv(COVID_AU_YOUGOV, index=False)
