from typing import Callable, Dict
from datetime import timedelta

import numpy

from autumn.models.covid_19.parameters import Sojourn
from autumn.tools.utils.time import EpochConverter
from autumn.models.covid_19.constants import BASE_DATETIME


def calc_compartment_periods(sojourn: Sojourn) -> Dict[str, float]:
    """
    Calculate compartment periods from the provided splits into early and late components of the exposed and active
    periods.
    """

    final_periods = {**sojourn.compartment_periods}
    for calc_period_name, calc_period_def in sojourn.compartment_periods_calculated.items():
        comp_final_periods = {
            f"{comp_name}_{calc_period_name}": calc_period_def.total_period * prop for
            comp_name, prop in calc_period_def.proportions.items()
        }
        final_periods.update(comp_final_periods)

    return final_periods


def get_seasonal_forcing(period: float, shift: float, forcing_magnitude: float, average_value: float) -> Callable:
    """
    Factory function to get a trigonometric/sinusoidal function (using cosine) to represent seasonal forcing of
    transmission in a model.
    Note that the time unit is not specified (as elsewhere in the repository), so the period is not assumed.
    :param period: float
        Time to complete an entire cycle of forcing
    :param shift: float
        Time at which the peak value will be reached
    :param forcing_magnitude: float
        Note that the amplitude is the total variation in transmission from trough to peak (consistent with the approach
            of others - e.g. Kissler et al. Science)
    :param average_value: float
        Average value of the function, mid-way between peak and trough values
    :return:
        The seasonal forcing function
    """

    amplitude = forcing_magnitude * average_value / 2.
    msg = f"Seasonal forcing parameters invalid, forcing magnitude: {forcing_magnitude}, contact rate: {average_value}"
    assert amplitude <= average_value, msg

    def seasonal_forcing(time, computed_values):
        return numpy.cos((time - shift) * 2. * numpy.pi / period) * amplitude + average_value

    return seasonal_forcing

def get_epoch_converter() -> EpochConverter:
    """Return an instance of a conversion class between the reference datetime and floating point offsets

    Returns:
        EpochConverter: The converter
    """
    return EpochConverter(BASE_DATETIME, timedelta(days=1))
