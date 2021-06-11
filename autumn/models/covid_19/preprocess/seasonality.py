import numpy


def get_seasonal_forcing(
    period: float, shift: float, seasonal_force_magnitude: float, average_value: float
):
    """
    Note that this is no longer being used. We considered it important for Victoria, but probably less so for the other
    applications. Reviewers of the Victoria paper recommended removing it, so we have done so, so now not used for any
    applications.

    Factory function to get a trigonometric/sinusoidal function (using cosine) to represent seasonal forcing of
    transmission in a model.
    Note that the time unit is not specified (as elsewhere in the repository), so the period is not assumed.

    :param period: float
        Time to complete an entire cycle of forcing
    :param shift: float
        Time at which the peak value will be reached
    :param seasonal_force_magnitude: float
        Note that the amplitude is the total variation in transmission from trough to peak (consistent with the approach
            of others - e.g. Kissler et al. Science)
    :param average_value: float
        Average value of the function, mid-way between peak and trough values
    :return:
        The seasonal forcing function
    """

    amplitude = seasonal_force_magnitude * average_value / 2.0
    assert (
        amplitude <= average_value
    ), "Seasonal forcing function will go negative based on submitted parameters"

    def seasonal_forcing(time):
        return numpy.cos((time - shift) * 2.0 * numpy.pi / period) * amplitude + average_value

    return seasonal_forcing


if __name__ == "__main__":
    period, shift, amplitude, average = 365.0, 173.0, 0.5, 0.04
    x_values = numpy.linspace(0.0, period, 20)
    forcing_function = get_seasonal_forcing(period, shift, amplitude, average)
    for i in x_values:
        print(f"date: {i}, \t value: {forcing_function(i)}")
