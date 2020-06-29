"""
Preprocess static social mixing data so it is included in the inputs database
"""
import os

import pandas as pd


from autumn import constants
from autumn.db import Database


MIXING_DIRPATH = os.path.join(constants.INPUT_DATA_PATH, "social-mixing")
LOCATIONS = ("all_locations", "home", "other_locations", "school", "work")
SHEET_NUMBERS = [("1", 0), ("2", None)]

SHEET_NAME_ISO3_MAP = {
    "Bolivia (Plurinational State of": "BOL",
    "Czech Republic": "CZE",
    "Hong Kong SAR, China": "HKG",
    "Lao People's Democratic Republi": "LAO",
    "Sao Tome and Principe ": "STP",
    "Taiwan": "TWN",
    "TFYR of Macedonia": "MKD",
    "United Kingdom of Great Britain": "GBR",
    "Venezuela (Bolivarian Republic ": "VEN",
}


def preprocess_social_mixing(input_db: Database, country_df):
    for location in LOCATIONS:
        for sheet_number, header_arg in SHEET_NUMBERS:
            sheet_name = f"MUestimates_{location}_{sheet_number}.xlsx"
            sheet_path = os.path.join(MIXING_DIRPATH, sheet_name)
            xl = pd.ExcelFile(sheet_path)
            sheet_names = xl.sheet_names
            iso3s = [get_iso3(n, country_df) for n in sheet_names]
            for idx, sheet_name in enumerate(sheet_names):
                iso3 = iso3s[idx]
                mix_df = pd.read_excel(xl, header=header_arg, sheet_name=sheet_name)
                if sheet_number == "2":
                    renames = {n - 1: f"X{n}" for n in range(1, 17)}
                    mix_df.rename(columns=renames, inplace=True)

                mix_df.insert(0, "location", [location for _ in range(len(mix_df))])
                mix_df.insert(0, "iso3", [iso3 for _ in range(len(mix_df))])
                input_db.dump_df("social_mixing", mix_df)


def get_iso3(sheet_name: str, country_df):
    try:
        return country_df[country_df["country"] == sheet_name]["iso3"].iloc[0]
    except IndexError:
        return SHEET_NAME_ISO3_MAP[sheet_name]
