"""
This file imports Google mobility data and saves it to disk as a CSV.
"""

from pathlib import Path
from autumn.settings import INPUT_DATA_PATH

INPUT_DATA_PATH = Path(INPUT_DATA_PATH)

COXS_VAC_DATA = (
    INPUT_DATA_PATH
    / "covid_bgd"
    / "1st phase daily data (Target age group 55 years and above).xlsx"
)


COXS_DATA = INPUT_DATA_PATH / "covid_bgd" / "COVID-19 Data for modelling.xlsx"

BGD_DATA_PATH = INPUT_DATA_PATH / "covid_bgd"

VACC_FILE = [
    BGD_DATA_PATH / vac_file
    for vac_file in list(BGD_DATA_PATH.glob("*"))
    if "DOSE" in vac_file.stem
]


def fetch_covid_bgd_data():
    pass
