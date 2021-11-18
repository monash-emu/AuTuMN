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

def get_lka_vac_coverage(age_group):
    times = [365, 395, 425, 455, 485, 515, 545, 575, 605, 635, 665, 695, 725]
    if int(age_group) < 15:
        coverage_values = [0.] * 13
    else:
        coverage_values = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.9, 0.9, 0.9]

    return times, coverage_values