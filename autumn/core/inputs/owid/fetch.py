"""
This file imports OWID data and saves it to disk as a CSV.
"""
import os

import pandas as pd

from autumn.settings import INPUT_DATA_PATH
from pathlib import Path
INPUT_DATA_PATH = Path(INPUT_DATA_PATH)

OWID_URL = (
    "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
)
OWID_DIRPATH = INPUT_DATA_PATH/ "owid"
OWID_CSV_PATH = OWID_DIRPATH/ "owid-covid-data.csv"


def fetch_owid_data():
    pd.read_csv(OWID_URL).to_csv(OWID_CSV_PATH)
    return None
