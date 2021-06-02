import numpy as np

from autumn.tools.inputs.database import get_input_db


def get_lka_testing_numbers():
    """
    Returns daily PCR test numbers for Sri lanka
    """

    input_db = get_input_db()
    df = input_db.query(
        "covid_lka",
        columns=["date_index", "PCR_tests_done"],
    )
    test_dates = df.date_index.to_numpy()
    test_values = df.PCR_tests_done.to_numpy()

    return test_dates, test_values
