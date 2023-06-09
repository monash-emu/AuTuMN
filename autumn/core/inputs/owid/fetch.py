"""
This file imports OWID data and saves it to disk as a CSV.
"""
from pathlib import Path

import pandas as pd
import yaml
import os
from autumn.settings import INPUT_DATA_PATH, PROJECTS_PATH

INPUT_DATA_PATH = Path(INPUT_DATA_PATH)

OWID_URL = (
    "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
)
OWID_DIRPATH = INPUT_DATA_PATH / "owid"
OWID_CSV_PATH = OWID_DIRPATH / "owid-covid-data.csv"

# Keep required countries due to large CSV filesize.
COUNTRY_FILTER = {
    "AU": "AUS",  # Australia
    "BD": "BGD",  # Bangladesh
    "BE": "BEL",  # Belgium
    "BT": "BTN",  # Bhutan
    "CN": "CHN",  # China
    "ES": "ESP",  # Spain
    "FR": "FRA",  # France
    "GB": "GBR",  # United Kingdom"
    "IT": "ITA",  # Italy
    "JP": "JPN",  # Japan
    "KR": "KOR",  # Korea
    "LK": "LKA",  # Sri Lanka
    "MM": "MMR",  # Myanmar
    "MN": "MNG",  # Mongolia
    "MY": "MYS",  # Malaysia
    "NZ": "NZL",  # New Zealand
    "PH": "PHL",  # Philippines
    "SE": "SWE",  # Sweden
    "SG": "SGP",  # Singapore
    "VN": "VNM",  # Vietnam
}


filter_iso3 = set(COUNTRY_FILTER.values())

# add countries from school closure project
school_country_source = os.path.join(PROJECTS_PATH, "sm_covid2", "common_school", "included_countries.yml")
school_country_dict = yaml.load(open(school_country_source), Loader=yaml.UnsafeLoader)
school_iso3s = set(school_country_dict['all'].keys())
filter_iso3.update(school_iso3s)

def fetch_owid_data() -> None:
    df = pd.read_csv(OWID_URL)
    df[df.iso_code.isin(filter_iso3)].to_csv(OWID_CSV_PATH)

    return None
