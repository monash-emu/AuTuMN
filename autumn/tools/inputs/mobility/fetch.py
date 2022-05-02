"""
This file imports Google mobility data and saves it to disk as a CSV.
"""
import os

import pandas as pd

from autumn.settings import INPUT_DATA_PATH

GOOGLE_MOBILITY_URL = (
    "https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv"
)
MOBILITY_DIRPATH = os.path.join(INPUT_DATA_PATH, "mobility")
MOBILITY_CSV_PATH = os.path.join(MOBILITY_DIRPATH, "Google_Mobility_Report.csv")
VNM_CSV_PATH = os.path.join(MOBILITY_DIRPATH, "VNM_mobility.csv")


FB_MOVEMENT_2021 = os.path.join(MOBILITY_DIRPATH, "movement-range-2021.txt")
FB_MOVEMENT_2022 = [
    os.path.join(MOBILITY_DIRPATH, file)
    for file in os.listdir(MOBILITY_DIRPATH)
    if file.startswith("movement-range-2022")
][0]

# Remove some countries due to large CSV filesize.
COUNTRY_FILTER = {
    "BR",
    "US",
    "AR",
    "IN",
    "CO",
    "CA",
    "EC",
    "PL",
    "TR",
    "RO",
    "NG",
    "PE",
    "BG",
    "SL",
    "GT",
    "CZ",
    "SK",
    "CL",
    "PT",
}


def fetch_mobility_data():
    df = pd.read_csv(GOOGLE_MOBILITY_URL)
    df[~df.country_region_code.isin(COUNTRY_FILTER)].to_csv(MOBILITY_CSV_PATH)
