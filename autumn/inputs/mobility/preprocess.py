import pandas as pd

from autumn.db import Database

from .fetch import MOBILITY_CSV_PATH

NAN = float("nan")
MOBILITY_SUFFIX = "_percent_change_from_baseline"


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


def preprocess_mobility(input_db: Database, country_df):
    """
    Read Google Mobility data from CSV into input database
    """
    mob_df = pd.read_csv(MOBILITY_CSV_PATH)

    # Drop all sub-region 2 data, too detailed.
    mob_df = mob_df[mob_df["sub_region_2"].isnull()]

    mob_cols = [c for c in mob_df.columns if c.endswith(MOBILITY_SUFFIX)]
    for c in mob_cols:
        # Convert percent values to decimal: 1.0 being no change.
        # Replace NA values with 1 - no change from baseline.
        mob_df[c] = mob_df[c].fillna(0)
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


def get_iso3(country_name: str, country_df):
    try:
        return country_df[country_df["country"] == country_name]["iso3"].iloc[0]
    except IndexError:
        return COUNTRY_NAME_ISO3_MAP[country_name]
