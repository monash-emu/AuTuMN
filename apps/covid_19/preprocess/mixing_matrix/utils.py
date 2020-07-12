"""
Utilities for helping build a mixing matrix
"""
from datetime import date, datetime

# Base date used to calculate mixing matrix times.
BASE_DATE = date(2019, 12, 31)
BASE_DATETIME = datetime(2019, 12, 31, 0, 0, 0)


def get_total_contact_rates_by_age(mixing_matrix, direction="horizontal"):
    """
    Sum the contact-rates by age group
    :param mixing_matrix: the input mixing matrix
    :param direction: either 'horizontal' (infectee's perspective) or 'vertical' (infector's perspective)
    :return: dict
        keys are the age categories and values are the aggregated contact rates
    """
    assert direction in [
        "horizontal",
        "vertical",
    ], "direction should be in ['horizontal', 'vertical']"
    aggregated_contact_rates = {}
    for i in range(16):
        if direction == "horizontal":
            aggregated_contact_rates[str(5 * i)] = mixing_matrix[i, :].sum()
        else:
            aggregated_contact_rates[str(5 * i)] = mixing_matrix[:, i].sum()
    return aggregated_contact_rates

