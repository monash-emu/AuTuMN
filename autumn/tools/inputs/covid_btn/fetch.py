"""
This file imports testing data 
"""
import os

from autumn.settings import INPUT_DATA_PATH


COVID_BTN_DIRPATH = os.path.join(INPUT_DATA_PATH, "covid_btn")
COVID_BTN_TEST_PATH = os.path.join(COVID_BTN_DIRPATH, "Testing numbers.xlsx")
COVID_BTN_VAC_PATH = os.path.join(COVID_BTN_DIRPATH, "Vaccination.xlsx")


def fetch_covid_btn_data():
    pass