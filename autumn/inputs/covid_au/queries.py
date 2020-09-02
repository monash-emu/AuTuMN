import numpy as np

from autumn.inputs.database import get_input_db
from autumn.tool_kit.utils import apply_moving_average

from datetime import date, datetime

COVID_BASE_DATE = date(2019, 12, 31)
COVID_BASE_DATETIME = datetime(2019, 12, 31, 0, 0, 0)


def get_vic_testing_numbers():
    """
    Returns 7-day moving average of number of tests administered in Victoria.
    """
    input_db = get_input_db()
    df = input_db.query("covid_au", column=["date", "tests"], conditions=[f"state_abbrev='VIC'"])
    date_str_to_int = lambda s: (datetime.strptime(s, "%Y-%m-%d") - COVID_BASE_DATETIME).days
    test_dates = df.date.apply(date_str_to_int).to_numpy()
    test_values = df.tests.to_numpy()
    epsilon = 1e-6  # A really tiny number to avoid having any zeros
    avg_vals = np.array(apply_moving_average(test_values, 7)) + epsilon
    return test_dates, avg_vals
