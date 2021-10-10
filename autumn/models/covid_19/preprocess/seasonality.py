import numpy
from typing import Callable


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
