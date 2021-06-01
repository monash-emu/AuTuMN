"""
This file imports OWID data and saves it to disk as a CSV.
"""
import os

import pandas as pd

from autumn.settings import INPUT_DATA_PATH

OWID_URL = (
    "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
)
OWID_DIRPATH = os.path.join(INPUT_DATA_PATH, "owid")
OWID_CSV_PATH = os.path.join(OWID_DIRPATH, "owid-covid-data.csv")


def fetch_owid_data():
    df = pd.read_csv(OWID_URL).to_csv(OWID_CSV_PATH)
