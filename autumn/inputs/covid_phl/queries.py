import numpy as np

from autumn.inputs.database import get_input_db
from autumn.utils.utils import apply_moving_average


def get_phl_subregion_testing_numbers(region):
    """
    Returns 7-day moving average of number of tests administered in Philippines & sub regions.
    """

    input_db = get_input_db()
    df = input_db.query(
        "covid_phl",
        columns=["date_index", "daily_output_unique_individuals"],
        conditions={"facility_name": region},
    )
    test_dates = df.date_index.to_numpy()
    test_values = df.daily_output_unique_individuals.to_numpy()
    epsilon = 1e-6  # A really tiny number to avoid having any zeros
    avg_vals = np.array(apply_moving_average(test_values, 7)) + epsilon
    return test_dates, avg_vals
