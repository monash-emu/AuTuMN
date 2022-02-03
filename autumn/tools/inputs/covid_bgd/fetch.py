"""
This file imports Google mobility data and saves it to disk as a CSV.
"""
import os


from autumn.settings import INPUT_DATA_PATH

COXS_VAC_DATA = os.path.join(
    INPUT_DATA_PATH, "covid_bgd", "1st phase daily data (Target age group 55 years and above).xlsx"
)

COXS_DATA = os.path.join(INPUT_DATA_PATH, "covid_bgd", "COVID-19 Data for modelling.xlsx")

BGD_DATA_PATH = os.path.join(INPUT_DATA_PATH, "covid_bgd")

VACC_FILE = [
    os.path.join(BGD_DATA_PATH, vac_file)
    for vac_file in os.listdir(BGD_DATA_PATH)
    if "DOSE" in vac_file
]


def fetch_covid_bgd_data():
    pass
