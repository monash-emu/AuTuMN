"""
This file imports Google mobility data and saves it to disk as a CSV.
"""
import os
import pandas as pd

from autumn import constants

GOOGLE_MOBILITY_URL = "https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv"
MOBILITY_DIRPATH = os.path.join(constants.INPUT_DATA_PATH, "mobility")
MOBILITY_CSV_PATH = os.path.join(MOBILITY_DIRPATH, "Google_Mobility_Report.csv")

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
}


def fetch_mobility_data():
    df = pd.read_csv(GOOGLE_MOBILITY_URL)
    df[
        (df.country_region_code != "BR")
        & (df.country_region_code != "US")
        & (df.country_region_code != "AR")
        & (df.country_region_code != "IN")
        & (df.country_region_code != "CO")
        & (df.country_region_code != "CA")
        & (df.country_region_code != "EC")
        & (df.country_region_code != "PL")
        & (df.country_region_code != "TR")
        & (df.country_region_code != "RO")
        & (df.country_region_code != "NG")
        & (df.country_region_code != "PE")
        & (df.country_region_code != "BG")
        & (df.country_region_code != "SL")
        & (df.country_region_code != "GT")
        & (df.country_region_code != "CZ")
        & (df.country_region_code != "SK")
        & (df.country_region_code != "CL")
        & (df.country_region_code != "PT")
    ].to_csv(MOBILITY_CSV_PATH)
