import numpy as np

from autumn.tools.inputs.database import get_input_db


def get_lka_testing_numbers():
    """
    Returns daily PCR test numbers for Sri lanka
    """

    input_db = get_input_db()
    df = input_db.query(
        "covid_lka",
        columns=["date_index", "Sri_Lanka_PCR_tests_done"],
    )
    df.dropna(how="any", inplace=True)
    test_dates = df.date_index.to_numpy()
    test_values = df.Sri_Lanka_PCR_tests_done.to_numpy()

    return test_dates, test_values
