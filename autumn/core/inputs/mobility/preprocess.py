import os

import pandas as pd

from autumn.core.db import Database
from autumn.settings import INPUT_DATA_PATH
from autumn.core.utils.utils import create_date_index
from autumn.settings.constants import COVID_BASE_DATETIME

from .fetch import (
    MOBILITY_CSV_PATH,
    VNM_CSV_PATH,
    FB_MOVEMENT_2021,
    FB_MOVEMENT_2022,
)

NAN = float("nan")
MOBILITY_SUFFIX = "_percent_change_from_baseline"
MOBILITY_DIRPATH = os.path.join(INPUT_DATA_PATH, "mobility")

DHHS_LGA_TO_CLUSTER = os.path.join(
    MOBILITY_DIRPATH, "LGA to Cluster mapping dictionary with proportions.csv"
)

DHHS_LGA_TO_HSP = os.path.join(INPUT_DATA_PATH, "covid_au", "LGA_HSP map_v2.csv")

MOBILITY_LGA_PATH = (
    DHHS_LGA_TO_HSP  # Swap with DHHS_LGA_TO_CLUSTER to obtain previous mapping
)


COUNTRY_NAME_ISO3_MAP = {
    "Bolivia": "BOL",
    "The Bahamas": "BHS",
    "Côte d'Ivoire": "CIV",
    "Cape Verde": "CPV",
    "Hong Kong": "HKG",
    "South Korea": "KOR",
    "Laos": "LAO",
    "Moldova": "MDA",
    "Myanmar (Burma)": "MMR",
    "Russia": "RUS",
    "Taiwan": "TWN",
    "Tanzania": "TZA",
    "United States": "USA",
    "Venezuela": "VEN",
    "Vietnam": "VNM",
}

VIC_LGA_MAP = {
    "Alpine (S)": "Alpine Shire",
    "Ararat (RC)": None,
    "Ballarat (C)": "City of Ballarat",
    "Banyule (C)": "City of Banyule",
    "Bass Coast (S)": "Bass Coast Shire",
    "Baw Baw (S)": "Baw Baw Shire",
    "Bayside (C)": "City of Bayside",
    "Benalla (RC)": "Benalla Rural City",
    "Boroondara (C)": "City of Boroondara",
    "Brimbank (C)": "City of Brimbank",
    "Buloke (S)": None,
    "Campaspe (S)": "Campaspe Shire",
    "Cardinia (S)": "Cardinia Shire",
    "Casey (C)": "City of Casey",
    "Central Goldfields (S)": "Central Goldfields Shire",
    "Colac-Otway (S)": "Colac Otway Shire",
    "Corangamite (S)": "Corangamite Shire",
    "Darebin (C)": "City of Darebin",
    "East Gippsland (S)": "East Gippsland Shire",
    "Frankston (C)": "City of Frankston",
    "Gannawarra (S)": None,
    "Glen Eira (C)": "City of Glen Eira",
    "Glenelg (S)": "Glenelg Shire",
    "Golden Plains (S)": "Golden Plains Shire",
    "Greater Bendigo (C)": "Greater Bendigo City",
    "Greater Dandenong (C)": "City of Greater Dandenong",
    "Greater Geelong (C)": "Greater Geelong City",
    "Greater Shepparton (C)": "Greater Shepparton City",
    "Hepburn (S)": "Hepburn Shire",
    "Hindmarsh (S)": None,
    "Hobsons Bay (C)": "City of Hobsons Bay",
    "Horsham (RC)": "Horsham Rural City",
    "Hume (C)": "City of Hume",
    "Indigo (S)": "Indigo Shire",
    "Kingston (C)": "City of Kingston",
    "Knox (C)": "City of Knox",
    "Latrobe (C)": "Latrobe City",
    "Loddon (S)": None,
    "Macedon Ranges (S)": "Macedon Ranges Shire",
    "Manningham (C)": "City of Manningham",
    "Mansfield (S)": None,
    "Maribyrnong (C)": "City of Maribyrnong",
    "Maroondah (C)": "Maroondah City",
    "Melbourne (C)": "City of Melbourne",
    "Melton (S)": "Melton City",
    "Mildura (RC)": "Mildura Rural City",
    "Mitchell (S)": "Mitchell Shire",
    "Moira (S)": "Moira Shire",
    "Monash (C)": "City of Monash",
    "Moonee Valley (C)": "City of Moonee Valley",
    "Moorabool (S)": "Moorabool Shire",
    "Moreland (C)": "City of Moreland",
    "Mornington Peninsula (S)": "Shire of Mornington Peninsula",
    "Mount Alexander (S)": "Mount Alexander Shire",
    "Moyne (S)": "Moyne Shire",
    "Murrindindi (S)": "Murrindindi Shire",
    "Nillumbik (S)": "Shire of Nillumbik",
    "Northern Grampians (S)": "Northern Grampians Shire",
    "Port Phillip (C)": "Port Phillip City",
    "Pyrenees (S)": None,
    "Queenscliffe (B)": None,
    "South Gippsland (S)": "South Gippsland Shire",
    "Southern Grampians (S)": "Southern Grampians Shire",
    "Stonnington (C)": "City of Stonnington",
    "Strathbogie (S)": None,
    "Surf Coast (S)": "Surf Coast Shire",
    "Swan Hill (RC)": "Swan Hill Rural City",
    "Towong (S)": None,
    "Wangaratta (RC)": "Wangaratta Rural City",
    "Warrnambool (C)": "City of Warrnambool",
    "Wellington (S)": "Wellington Shire",
    "West Wimmera (S)": None,
    "Whitehorse (C)": "Whitehorse City",
    "Whittlesea (C)": "City of Whittlesea",
    "Wodonga (RC)": "City of Wodonga",
    "Wyndham (C)": "City of Wyndham",
    "Yarra (C)": "City of Yarra",
    "Yarra Ranges (S)": "Yarra Ranges Shire",
    "Yarriambiack (S)": None,
}


def preprocess_mobility(input_db: Database, country_df):
    """
    Read Google Mobility data from CSV into input database
    """
    mob_df = pd.read_csv(MOBILITY_CSV_PATH)

    # Drop all sub-region 2 data, too detailed.
    major_region_mask = mob_df["sub_region_2"].isnull() & mob_df["metro_area"].isnull()
    davao_mask = mob_df.metro_area == "Davao City Metropolitan Area"
    mob_df = mob_df[major_region_mask | davao_mask].copy()

    # These two regions are the same
    mob_df.loc[
        (mob_df.sub_region_1 == "National Capital Region"), "sub_region_1"
    ] = "Metro Manila"
    mob_df.loc[
        (mob_df.metro_area == "Davao City Metropolitan Area"), "sub_region_1"
    ] = "Davao City"
    mob_df.loc[
        (mob_df.sub_region_1 == "Federal Territory of Kuala Lumpur"), "sub_region_1"
    ] = "Kuala Lumpur"

    # Read and append mobility predictions for Vietnam
    vnm_mob = pd.read_csv(VNM_CSV_PATH)
    mob_df = mob_df.merge(
        vnm_mob, on=["date", "country_region", "sub_region_1"], how="left"
    )
    col_str = "workplaces_percent_change_from_baseline"
    mob_df.loc[
        (mob_df["country_region"] == "Vietnam") & (mob_df[f"{col_str}_x"].isna()),
        f"{col_str}_x",
    ] = mob_df[f"{col_str}_y"]
    mob_df = mob_df.drop(columns=f"{col_str}_y")
    mob_df.rename(columns={f"{col_str}_x": col_str}, inplace=True)

    # Drop all rows that have NA values in 1 or more mobility columns.
    mob_cols = [c for c in mob_df.columns if c.endswith(MOBILITY_SUFFIX)]
    mask = False
    for c in mob_cols:
        mask = mask | mob_df[c].isnull()

    mob_df = mob_df[~mask].copy()
    for c in mob_cols:
        # Convert percent values to decimal: 1.0 being no change.
        mob_df[c] = mob_df[c].apply(lambda x: 1 + x / 100)

    # Drop unused columns, rename kept columns
    cols_to_keep = [*mob_cols, "country_region", "sub_region_1", "date"]
    cols_to_drop = [c for c in mob_df.columns if not c in cols_to_keep]
    mob_df = mob_df.drop(columns=cols_to_drop)
    mob_col_rename = {c: c.replace(MOBILITY_SUFFIX, "") for c in mob_cols}
    mob_df.rename(columns={**mob_col_rename, "sub_region_1": "region"}, inplace=True)

    # Convert countries to ISO3
    countries = mob_df["country_region"].unique().tolist()
    iso3s = {c: get_iso3(c, country_df) for c in countries}
    iso3_series = mob_df["country_region"].apply(lambda c: iso3s[c])
    mob_df.insert(0, "iso3", iso3_series)
    mob_df = mob_df.drop(columns=["country_region"])

    mob_df = mob_df.sort_values(["iso3", "region", "date"])
    input_db.dump_df("mobility", mob_df)

    # Facebook movement data
    df_list = []
    iso_filter = {"AUS", "PHL", "MYS", "VNM", "LKA", "IDN", "MYN", "BGD", "BTN"}
    for file in {FB_MOVEMENT_2021, FB_MOVEMENT_2022}:
        df = pd.read_csv(file, "\t")
        df_list.append(df)

    df = pd.concat(df_list)
    df = df[df["country"].isin(iso_filter)]
    df = df.sort_values(["country", "ds", "polygon_id"]).reset_index(drop=True)
    df = create_date_index(COVID_BASE_DATETIME, df, "ds")
    input_db.dump_df("movement", df)


def get_iso3(country_name: str, country_df):
    try:
        return country_df[country_df["country"] == country_name]["iso3"].iloc[0]
    except IndexError:
        return COUNTRY_NAME_ISO3_MAP[country_name]

