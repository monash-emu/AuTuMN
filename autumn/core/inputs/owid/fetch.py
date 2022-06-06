"""
This file imports OWID data and saves it to disk as a CSV.
"""
from pathlib import Path

import pandas as pd
from autumn.settings import INPUT_DATA_PATH

INPUT_DATA_PATH = Path(INPUT_DATA_PATH)

OWID_URL = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
OWID_DIRPATH = INPUT_DATA_PATH / "owid"
OWID_CSV_PATH = OWID_DIRPATH / "owid-covid-data.csv"

# Keep required countries due to large CSV filesize.
COUNTRY_FILTER = {
    "AU": "AUS",  # Australia
    "BT": "BTN",  # Bhutan
    "BD": "BGD",  # Bangladesh
    "BE": "BEL",  # Belgium
    "ES": "ESP",  # Spain
    "FR": "FRA",  # France
    "GB": "GBR",  # United Kingdom"
    "IT": "ITA",  # Italy
    "LK": "LKA",  # Sri Lanka
    "MM": "MMR",  # Myanmar
    "MY": "MYS",  # Malaysia
    "PH": "PHL",  # Philippines
    "SE": "SWE",  # Sweden
    "VN": "VNM",  # Vietnam
}

filter_iso3 = set(COUNTRY_FILTER.values())


def fetch_owid_data() -> None:
    df = pd.read_csv(OWID_URL)
    df[df.iso_code.isin(filter_iso3)].to_csv(OWID_CSV_PATH)

    return None
