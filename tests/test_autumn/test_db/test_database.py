from autumn.coreinputs.database import get_input_db

db = get_input_db()


def test_database__with_read_table__expect_table_df():
    """
    Ensure we can read a table from the input db as a dataframe.
    """
    result_df = db.query(table_name="countries")
    assert len(result_df.columns) == 3  # Number of columns
    eth_df = result_df[result_df["country"] == "Ethiopia"]
    assert eth_df["iso3"].iloc[0] == "ETH"


def test_database__with_conditions__expect_filtered_df():
    """
    Ensure we can read a filtered table from the input db as a dataframe.
    """
    result_df = db.query(
        table_name="countries",
        conditions={"country": "Ethiopia"},
    )
    assert len(result_df) == 1  # Number of rows
    assert len(result_df.columns) == 3  # Number of columns
    assert result_df["iso3"].iloc[0] == "ETH"


def test_database__with_conditions_and_column__expect_filtered_df():
    """
    Ensure we can read a single column from a filtered table from the input db as a dataframe.
    """
    result_df = db.query(
        table_name="countries",
        columns=["iso3"],
        conditions={"country": "Ethiopia"},
    )
    assert len(result_df) == 1  # Number of rows
    assert len(result_df.columns) == 1  # Number of columns
    assert result_df["iso3"].iloc[0] == "ETH"
