"""
This file is a placeholder. The data.csv must be manually downloaded 
from [Link](https://covid-19.health.gov.lk/dhis-web-dashboard/)
"""
from autumn.settings import INPUT_DATA_PATH
from pathlib import Path

INPUT_DATA_PATH = Path(INPUT_DATA_PATH)

COVID_LKA_CSV = INPUT_DATA_PATH / "covid_lka" / "data.csv"
COVID_LKA_2021_CSV = INPUT_DATA_PATH / "covid_lka" / "data_2021.csv"


def fetch_covid_lka_data():
    pass
