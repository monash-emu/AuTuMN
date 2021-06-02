"""
This file imports testing data from google drive and saves it to disk as a CSV.
See Readme.md \data\inputs\covid_phl on how to update DATA_URL
"""
import os
from autumn.settings import INPUT_DATA_PATH


COVID_LKA_CSV = os.path.join(INPUT_DATA_PATH,"covid_lka","data.csv")


def fetch_covid_lka_data():
    pass

