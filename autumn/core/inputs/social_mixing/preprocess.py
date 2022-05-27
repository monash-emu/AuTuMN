"""
Preprocess static social mixing data so it is included in the inputs database
"""
import os

import pandas as pd

from autumn.core.db import Database
from autumn.settings import INPUT_DATA_PATH
from autumn.core.inputs.social_mixing.constants import LOCATIONS

MIXING_DIRPATH = os.path.join(INPUT_DATA_PATH, "social-mixing")
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

    # Next gen social mixing
    original_mm = input_db.query("social_mixing")

    df = pd.read_csv(os.path.join(MIXING_DIRPATH, "synthetic_contacts_2020.csv"))
    df = df[df.setting == "overall"]
    df.drop(columns="setting", inplace=True)
    df.replace(
        {
            "0 to 4": "00 to 04",
            "5 to 9": "05 to 09",
            "all": "all_locations",
            "others": "other_locations",
        },
        inplace=True,
    )

    # The contactor is in j (columns) and the contactee is in i (rows)
    df = df.pivot_table(
        index=["iso3c", "location_contact", "age_cotactee"],
        columns="age_contactor",
        values="mean_number_of_contacts",
    )
    df = df.reset_index()
    df.drop(columns="age_cotactee", inplace=True)

    cols = list(df.columns[2:])
    new_col = ["X" + str(x) for x in range(1, len(cols) + 1)]
    replace_col = dict(zip(cols, new_col))
    df.rename(columns=replace_col, inplace=True)
    df.rename(columns={"iso3c": "iso3", "location_contact": "location"}, inplace=True)

    iso3_diff = set(original_mm.iso3).difference(df.iso3)
    iso3_mask = original_mm.iso3.isin(iso3_diff)
    df = df.append(original_mm[iso3_mask], ignore_index=True)

    input_db.dump_df("social_mixing_2020", df)


def get_iso3(sheet_name: str, country_df):
    try:
        return country_df[country_df["country"] == sheet_name]["iso3"].iloc[0]
    except IndexError:
        return SHEET_NAME_ISO3_MAP[sheet_name]
