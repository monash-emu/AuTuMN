import os

from autumn import constants
from autumn.db import Database

INPUT_DB_PATH = os.path.join(constants.DATA_PATH, "inputs.db")


def test_database__with_read_table__expect_table_df():
    """
    Ensure we can read a table from the input db as a dataframe.
    """
    db = Database(INPUT_DB_PATH)
    result_df = db.db_query(table_name="un_iso3_map")
    assert len(result_df) == 289  # Number of rows
    assert len(result_df.columns) == 37  # Number of columns
    eth_df = result_df[result_df["Region, subregion, country or area*"] == "Ethiopia"]
    assert eth_df["ISO3 Alpha-code"].iloc[0] == "ETH"


def test_database__with_conditions__expect_filtered_df():
    """
    Ensure we can read a filtered table from the input db as a dataframe.
    """
    db = Database(INPUT_DB_PATH)
    result_df = db.db_query(
        table_name="un_iso3_map", conditions=['"Region, subregion, country or area*"="Ethiopia"'],
    )
    assert len(result_df) == 1  # Number of rows
    assert len(result_df.columns) == 37  # Number of columns
    assert result_df["ISO3 Alpha-code"].iloc[0] == "ETH"


def test_database__with_conditions_and_column__expect_filtered_df():
    """
    Ensure we can read a single column from a filtered table from the input db as a dataframe.
    """
    db = Database(INPUT_DB_PATH)
    result_df = db.db_query(
        table_name="un_iso3_map",
        column='"ISO3 Alpha-code"',
        conditions=['"Region, subregion, country or area*"="Ethiopia"'],
    )
    assert len(result_df) == 1  # Number of rows
    assert len(result_df.columns) == 1  # Number of columns
    assert result_df["ISO3 Alpha-code"].iloc[0] == "ETH"
