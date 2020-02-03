"""
Methods for creating an input database
"""
import os
import time
import glob

import pandas as pd

from .. import constants
from .database import Database


def build_input_database():
    """
    Builds an input database from source Excel spreadsheets and stores it in the data directory.
    """
    # Load input database, where we will store the data.
    db_name = get_new_database_name()
    database = Database(db_name)

    # Load Excel sheets into the database.
    excel_glob = os.path.join(constants.EXCEL_PATH, "*.xlsx")
    excel_sheets = glob.glob(excel_glob)
    for file_path in excel_sheets:
        filename = os.path.basename(file_path)
        header_row = HEADERS_LOOKUP[filename] if filename in HEADERS_LOOKUP else 0
        data_title = OUTPUT_NAME[filename] if filename in OUTPUT_NAME else filename
        file_df = pd.read_excel(
            pd.ExcelFile(file_path),
            header=header_row,
            index_col=1,
            sheet_name=TAB_OF_INTEREST[filename],
        )
        print("Reading '%s' tab of '%s' file" % (TAB_OF_INTEREST[filename], filename))
        file_df.to_sql(data_title, con=database.engine, if_exists="replace")

    # Load CSV files into the database
    csv_glob = os.path.join(constants.EXCEL_PATH, "*.csv")
    csv_sheets = glob.glob(csv_glob)
    for file_path in csv_sheets:
        file_title = os.path.basename(file_path).split(".")[0]
        file_df = pd.read_csv(file_path)
        print("Reading '%s' file" % (file_path))

        file_df.to_sql(file_title, con=database.engine, if_exists="replace")

    # Add mapped ISO3 code tables that only contain the UN country code
    table_names = ["crude_birth_rate", "absolute_deaths", "total_population"]
    for table_name in table_names:
        print("Creating country code mapped database for", table_name)
        # Create dictionary structure to map from un three numeric digit codes to iso3 three alphabetical digit codes.
        map_df = database.db_query(table_name="un_iso3_map")[
            ["Location code", "ISO3 Alpha-code"]
        ].dropna()
        table_df = database.db_query(table_name=table_name)
        table_with_iso = pd.merge(
            table_df, map_df, left_on="Country code", right_on="Location code"
        )
        # Rename columns to avoid using spaces.
        table_with_iso.rename(columns={"ISO3 Alpha-code": "iso3"}, inplace=True)
        # Remove index column to avoid creating duplicates.
        if "Index" in table_with_iso.columns:
            table_with_iso = table_with_iso.drop(columns=["Index"])

        # Create a new 'mapped' database structure
        table_with_iso.to_sql(table_name + "_mapped", con=database.engine, if_exists="replace")

    return database


def get_new_database_name():
    """
    Get a timestamped name for the new database.
    """
    timestamp = int(time.time())
    db_name = f"inputs.{timestamp}.db"
    return os.path.join(constants.DATA_PATH, db_name)


# Mappings for Excel data that is used to populate the input database.
HEADERS_LOOKUP = {
    "WPP2019_FERT_F03_CRUDE_BIRTH_RATE.xlsx": 16,
    "WPP2019_F01_LOCATIONS.xlsx": 16,
    "WPP2019_MORT_F04_1_DEATHS_BY_AGE_BOTH_SEXES.xlsx": 16,
    "WPP2019_POP_F07_1_POPULATION_BY_AGE_BOTH_SEXES.xlsx": 16,
    "life_expectancy_2015.xlsx": 3,
    "rate_birth_2015.xlsx": 3,
}

TAB_OF_INTEREST = {
    "WPP2019_FERT_F03_CRUDE_BIRTH_RATE.xlsx": "ESTIMATES",
    "WPP2019_MORT_F04_1_DEATHS_BY_AGE_BOTH_SEXES.xlsx": "ESTIMATES",
    "WPP2019_POP_F07_1_POPULATION_BY_AGE_BOTH_SEXES.xlsx": "ESTIMATES",
    "WPP2019_F01_LOCATIONS.xlsx": "Location",
    "coverage_estimates_series.xlsx": "BCG",
    "gtb_2015.xlsx": "gtb_2015",
    "gtb_2016.xlsx": "gtb_2016",
    "life_expectancy_2015.xlsx": "life_expectancy_2015",
    "rate_birth_2015.xlsx": "rate_birth_2015",
}

OUTPUT_NAME = {
    "WPP2019_FERT_F03_CRUDE_BIRTH_RATE.xlsx": "crude_birth_rate",
    "WPP2019_MORT_F04_1_DEATHS_BY_AGE_BOTH_SEXES.xlsx": "absolute_deaths",
    "WPP2019_POP_F07_1_POPULATION_BY_AGE_BOTH_SEXES.xlsx": "total_population",
    "WPP2019_F01_LOCATIONS.xlsx": "un_iso3_map",
    "coverage_estimates_series.xlsx": "bcg",
    "gtb_2015.xlsx": "gtb_2015",
    "gtb_2016.xlsx": "gtb_2016",
    "life_expectancy_2015.xlsx": "life_expectancy_2015",
    "rate_birth_2015.xlsx": "rate_birth_2015",
}
