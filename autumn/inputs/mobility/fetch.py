"""
This file imports Google mobility data and saves it to disk as a CSV.
"""
import os
import pandas as pd

from autumn import constants

GOOGLE_MOBILITY_URL = "https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv"
MOBILITY_DIRPATH = os.path.join(constants.INPUT_DATA_PATH, "mobility")
MOBILITY_CSV_PATH = os.path.join(MOBILITY_DIRPATH, "Google_Mobility_Report.csv")


def fetch_mobility_data():
    df = pd.read_csv(GOOGLE_MOBILITY_URL)
    df[
        (df.country_region_code != "BR")
        & (df.country_region_code != "US")
        & (df.country_region_code != "AR")
    ].to_csv(MOBILITY_CSV_PATH)
