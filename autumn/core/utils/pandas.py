"""Utilities for interacting with Pandas objects
"""

import re
from typing import List
import pandas as pd
import numpy as np


def _pdfilt(df, fstr):
    m = re.match("(\S*)\s*(<=|>=|==|>|<)\s*(.*)", fstr)
    column, op, value = m.groups()
    op_table = {"<": "lt", "<=": "ge", ">": "gt", ">=": "ge", "==": "eq"}
    try:
        # Assume most things are floating point values; fall back to string otherwise
        value = float(value)
    except:
        pass

    return df[df[column].__getattribute__(op_table[op])(value)]


def pdfilt(df: pd.DataFrame, filters: List[str]) -> pd.DataFrame:
    """Return a filtered DataFrame, filtered by strings of form
    'column op value'
    E.g
    'mycolumn > 21.5'
    'valuetype == notifications'

    Args:
        df (pd.DataFrame): DataFrame to filter
        filters (list): List of filter strings (or single string)

    Returns:
        [pd.DataFrame]: Filtered DataFrame
    """
    if isinstance(filters, str):
        filters = [filters]

    for f in filters:
        df = _pdfilt(df, f)
    return df


def increment_last_period(
    recency: int, 
    values: pd.Series,
) -> pd.Series:
    """
    Find the increase in a series of increasing values over
    a preceding period of a certain duration.

    Args:
        recency: How far to look back to find the increment
        values: The series of values to look back into
    Returns:
        Series for the increases over the preceding period
    """
    assert values.is_monotonic_increasing
    assert values.index.is_monotonic_increasing

    return values - pd.Series(
        np.interp(values.index - recency, values.index, values), 
        index=values.index
    )
