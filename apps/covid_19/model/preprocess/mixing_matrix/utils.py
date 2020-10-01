"""
Utilities for helping build a mixing matrix
"""
from typing import List
from datetime import date, datetime, timedelta

from apps.covid_19.constants import BASE_DATE, BASE_DATETIME


def date_to_days(dates: List[date]) -> List[int]:
    return datetime_to_days(dates_to_datetimes(dates))


def days_to_dates(days: List[int]) -> List[datetime]:
    return [BASE_DATE + timedelta(days=d) for d in days]


def dates_to_datetimes(dates: List[date]) -> List[datetime]:
    return [datetime(d.year, d.month, d.day) for d in dates]


def datetime_to_days(datetimes: List[datetime]) -> List[int]:
    return [(dt - BASE_DATETIME).days for dt in datetimes]


def days_to_datetime(days: List[int]) -> List[datetime]:
    return [BASE_DATETIME + timedelta(days=d) for d in days]


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
