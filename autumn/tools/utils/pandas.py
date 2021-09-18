"""Utilities for interacting with Pandas objects
"""

import re
from typing import List

import pandas as pd

def _pdfilt(df, fstr):
    m = re.match('(\S*)\s*(<=|>=|==|>|<)\s*(.*)', fstr)
    column, op, value = m.groups()
    op_table = {
        '<': 'lt',
        '<=': 'ge',
        '>': 'gt',
        '>=': 'ge',
        '==': 'eq'
    }
    return df[df[column].__getattribute__(op_table[op])(float(value))]

def pdfilt(df: pd.DataFrame, filters: List[str]) -> pd.DataFrame:
    """Return a filtered DataFrame, filtered by strings of form
    'column op value'
    E.g
    'mycolumn > 21.5'

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
