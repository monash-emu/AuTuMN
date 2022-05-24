"""
This file imports testing data 
"""


from autumn.settings import INPUT_DATA_PATH
from pathlib import Path
INPUT_DATA_PATH = Path(INPUT_DATA_PATH)

COVID_BTN_DIRPATH = INPUT_DATA_PATH / "covid_btn"
COVID_BTN_TEST_PATH = COVID_BTN_DIRPATH / "Testing numbers.xlsx"
COVID_BTN_VAC_PATH = COVID_BTN_DIRPATH / "Vaccination.xlsx"


def fetch_covid_btn_data():
    pass