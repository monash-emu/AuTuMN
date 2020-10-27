import numpy as np

from autumn.inputs.database import get_input_db
from autumn.tool_kit.utils import apply_moving_average


def get_phl_testing_numbers(region):
    """
    Returns 7-day moving average of number of tests administered in Philippines & sub regions.
    """

    input_db = get_input_db()
    if region not in {"central visayas", "metro manila", "central visayas"}:
        df = input_db.query("covid_phl", column=["date_index", "daily_output_unique_individuals"])
        df = df.groupby(["date_index"]).sum().reset_index()
    else:
        df = input_db.query(
            "covid_phl",
            column=["date_index", "daily_output_unique_individuals"],
            conditions=[f"facility_name='" + region + "'"],
        )

    test_dates = df.date_index.to_numpy()
    test_values = df.daily_output_unique_individuals.to_numpy()
    epsilon = 1e-6  # A really tiny number to avoid having any zeros
    avg_vals = np.array(apply_moving_average(test_values, 7)) + epsilon
    return test_dates, avg_vals
