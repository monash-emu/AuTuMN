from autumn.tools.inputs.database import get_input_db


def get_mmr_testing_numbers():
    """
    Returns daily PCR test numbers for Myanmar
    """

    input_db = get_input_db()
    df = input_db.query(
        "covid_mmr",
        columns=["date_index", "tests"],
    )

    df.dropna(how="any", inplace=True)

    test_dates = df.date_index.to_numpy()
    test_values = df.tests.to_numpy()

    return test_dates, test_values

