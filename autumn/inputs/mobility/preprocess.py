import pandas as pd
import os
from autumn.db import Database

from .fetch import MOBILITY_CSV_PATH
from autumn import constants

NAN = float("nan")
MOBILITY_SUFFIX = "_percent_change_from_baseline"
MOBILITY_DIRPATH = os.path.join(constants.INPUT_DATA_PATH, "mobility")
MOBILITY_LGA_PATH = os.path.join(MOBILITY_DIRPATH, "LGA to Cluster mapping dictionary with proportions.csv")

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
    "Alpine (S)" : "Alpine Shire",
    "Ararat (RC)" : "Victoria",
    "Ballarat (C)" : "Ballarat City",
    "Banyule (C)" : "Banyule City",
    "Bass Coast (S)" : "Bass Coast Shire",
    "Baw Baw (S)" : "Baw Baw Shire",
    "Bayside (C)" : "Bayside City",
    "Benalla (RC)" : "Benalla Rural City",
    "Boroondara (C)" : "Boroondara City",
    "Brimbank (C)" : "Brimbank City",
    "Buloke (S)" : "Victoria",
    "Campaspe (S)" : "Campaspe Shire",
    "Cardinia (S)" : "Cardinia Shire",
    "Casey (C)" : "City of Casey",
    "Central Goldfields (S)" : "Central Goldfields Shire",
    "Colac-Otway (S)" : "Colac Otway Shire",
    "Corangamite (S)" : "Corangamite Shire",
    "Darebin (C)" : "Darebin City",
    "East Gippsland (S)" : "East Gippsland Shire",
    "Frankston (C)" : "Frankston City",
    "Gannawarra (S)" : "Victoria",
    "Glen Eira (C)" : "Glen Eira City",
    "Glenelg (S)" : "Glenelg Shire",
    "Golden Plains (S)" : "Golden Plains Shire",
    "Greater Bendigo (C)" : "Greater Bendigo City",
    "Greater Dandenong (C)" : "Greater Dandenong City",
    "Greater Geelong (C)" : "Greater Geelong City",
    "Greater Shepparton (C)" : "Greater Shepparton City",
    "Hepburn (S)" : "Hepburn Shire",
    "Hindmarsh (S)" : "Victoria",
    "Hobsons Bay (C)" : "Hobsons Bay City",
    "Horsham (RC)" : "Horsham Rural City",
    "Hume (C)" : "City of Hume",
    "Indigo (S)" : "Indigo Shire",
    "Kingston (C)" : "Kingston City",
    "Knox (C)" : "City of Knox",
    "Latrobe (C)" : "Latrobe City",
    "Loddon (S)" : "Victoria",
    "Macedon Ranges (S)" : "Macedon Ranges Shire",
    "Manningham (C)" : "Manningham City",
    "Mansfield (S)" : "Victoria",
    "Maribyrnong (C)" : "City of Maribyrnong",
    "Maroondah (C)" : "Maroondah City",
    "Melbourne (C)" : "Melbourne City",
    "Melton (S)" : "Melton City",
    "Mildura (RC)" : "Mildura Rural City",
    "Mitchell (S)" : "Mitchell Shire",
    "Moira (S)" : "Moira Shire",
    "Monash (C)" : "Monash City",
    "Moonee Valley (C)" : "Moonee Valley City",
    "Moorabool (S)" : "Moorabool Shire",
    "Moreland (C)" : "Moorabool Shire",
    "Mornington Peninsula (S)" : "Shire of Mornington Peninsula",
    "Mount Alexander (S)" : "Mount Alexander Shire",
    "Moyne (S)" : "Moyne Shire",
    "Murrindindi (S)" : "Murrindindi Shire",
    "Nillumbik (S)" : "Nillumbik Shire",
    "Northern Grampians (S)" : "Northern Grampians Shire",
    "Port Phillip (C)" : "Port Phillip City",
    "Pyrenees (S)" : "Victoria",
    "Queenscliffe (B)" : "Victoria",
    "South Gippsland (S)" : "South Gippsland Shire",
    "Southern Grampians (S)" : "Southern Grampians Shire",
    "Stonnington (C)" : "Stonnington City",
    "Strathbogie (S)" : "Victoria",
    "Surf Coast (S)" : "Surf Coast Shire",
    "Swan Hill (RC)" : "Swan Hill Rural City",
    "Towong (S)" : "Victoria",
    "Wangaratta (RC)" : "Wangaratta Rural City",
    "Warrnambool (C)" : "Warrnambool City",
    "Wellington (S)" : "Wellington Shire",
    "West Wimmera (S)" : "Victoria",
    "Whitehorse (C)" : "Whitehorse City",
    "Whittlesea (C)" : "Whittlesea City",
    "Wodonga (RC)" : "Wodonga City",
    "Wyndham (C)" : "City of Wyndham",
    "Yarra (C)" : "City of Yarra",
    "Yarra Ranges (S)" : "Yarra Ranges Shire",
    "Yarriambiack (S)" : "Victoria"
    }


def preprocess_mobility(input_db: Database, country_df):
    """
    Read Google Mobility data from CSV into input database
    """
    mob_df = pd.read_csv(MOBILITY_CSV_PATH)

    
    dhhs_cluster_mobility = reshape_to_clusters(mob_df)

    # Drop all sub-region 2 data, too detailed.
    mob_df = mob_df[
                    (mob_df["sub_region_2"].isnull())
                    & (mob_df["metro_area"].isnull())
                    ]

    # These two regions are the same
    mob_df.loc[(mob_df.sub_region_1 == "National Capital Region"),"sub_region_1"] = "Metro Manila"

    mob_df = mob_df.append(dhhs_cluster_mobility)
         
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


def reshape_to_clusters(gm_df):

    # Before dropping sub_region_2 capture Victorian LGAs.
    gm_df = gm_df[
                 (gm_df.sub_region_1 == "Victoria") 
                 #& (gm_df.sub_region_2.notnull())
                 ]
    gm_df["sub_region_1"] = gm_df["sub_region_2"]
    gm_df.loc[(gm_df.sub_region_1.isnull()),"sub_region_1"] = "Victoria"
    
    lga_df = pd.read_csv(MOBILITY_LGA_PATH) 
    lga_df.replace({"lga_name":VIC_LGA_MAP}, inplace=True)
    lga_df = pd.merge(lga_df, gm_df, how="left", left_on="lga_name", right_on="sub_region_1")

    list_of_columns = ["retail_and_recreation_percent_change_from_baseline",
                 "grocery_and_pharmacy_percent_change_from_baseline",
                 "parks_percent_change_from_baseline",
                 "transit_stations_percent_change_from_baseline",
                 "workplaces_percent_change_from_baseline",
                 "residential_percent_change_from_baseline"]

    multiplied_columns = lga_df[list_of_columns].multiply(lga_df.proportion,  axis="index")
    cluster_df = lga_df[["cluster_name","date"]]
    
    cluster_df = pd.concat([cluster_df,multiplied_columns], axis=1, sort=False)
    cluster_df = cluster_df.groupby(["cluster_name","date"]).sum()
    cluster_df.reset_index(inplace=True)
    cluster_df["country_region"] = "Australia"

    cluster_df["cluster_name"] = cluster_df["cluster_name"].str.replace('South & East Metro','SOUTH_EAST_METRO')
    cluster_df["cluster_name"] = cluster_df["cluster_name"].str.replace('&','').str.replace(' ','_').str.upper()
    
    cluster_df.rename(columns={"cluster_name":"sub_region_1"}, inplace=True)
       
    return cluster_df


