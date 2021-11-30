"""
This file imports Google mobility data and saves it to disk as a CSV.
"""
import os

import pandas as pd

from autumn.settings import INPUT_DATA_PATH

# From covid19data.com.au GitHub https://github.com/M3IT/COVID-19_Data
DATA_URL = "https://raw.githubusercontent.com/M3IT/COVID-19_Data/master/Data/COVID_AU_state_daily_change.csv"
COVID_AU_DIRPATH = os.path.join(INPUT_DATA_PATH, "covid_au")
COVID_AU_CSV_PATH = os.path.join(COVID_AU_DIRPATH, "COVID_AU_state_daily_change.csv")

YOUGOV_URL = "https://github.com/YouGov-Data/covid-19-tracker/raw/master/data/australia.zip"
COVID_AU_YOUGOV = os.path.join(COVID_AU_DIRPATH, "yougov_australia.csv")

COVID_LGA_CSV_PATH = os.path.join(COVID_AU_DIRPATH, "lga_test.csv")
MOBILITY_DIRPATH = os.path.join(INPUT_DATA_PATH, "mobility")
MOBILITY_LGA_PATH = os.path.join(
    MOBILITY_DIRPATH, "LGA to Cluster mapping dictionary with proportions.csv"
)
COVID_VAC_COV_CSV = os.path.join(COVID_AU_DIRPATH, "vac_cov.csv")
COVID_VIDA_VAC_CSV = os.path.join(COVID_AU_DIRPATH, "vida_vac.secret.csv")
COVID_VIDA_POP_CSV = os.path.join(COVID_AU_DIRPATH, "vida_pop.csv")


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
