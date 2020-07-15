"""
Functions which can be used to transform dynamic mixing timeseries data
"""
from typing import List


def repeat_prev(prev_vals: List[float]):
    """
    Repeats the previous seen value again
    """
    return prev_vals[-1]


def add_to_prev(prev_vals: List[float], increment: float):
    """
    Add increment to previous
    """
    val = prev_vals[-1] + increment
    if val < 0:
        return 0
    else:
        return val


def add_to_prev_up_to_1(prev_vals: List[float], increment: float):
    """
    Add increment to previous
    """
    val = prev_vals[-1] + increment
    if val > 1:
        return 1
    elif val < 0:
        return 0
    else:
        return val


def scale_prev(prev_vals: List[float], fraction: float):
    """
    Apply a percentage to the previous value, saturating at zero
    """
    val = prev_vals[-1] * fraction
    if val < 0:
        return 0
    else:
        return val


def scale_prev_up_to_1(prev_vals: List[float], fraction: float):
    """
    Apply a percentage to the previous value, saturating at one or zero
    """
    val = prev_vals[-1] * fraction
    if val > 1:
        return 1
    elif val < 0:
        return 0
    else:
        return val
