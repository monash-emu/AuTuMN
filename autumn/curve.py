# -*- coding: utf-8 -*-

"""

Sigmoidal function to generate cost coverage curves and historiacl
input curves.

"""


from math import exp


def make_sigmoidal_curve(y_low=0, y_high=1.0, x_start=0, x_inflect=0.5, multiplier=1.):
    """
    Args:
        y_low: lowest y value
        y_high: highest y value
        x_inflect: inflection point of graph along the x-axis
        multiplier: if 1, slope at x_inflect goes to (0, y_low), larger
                    values makes it steeper

    Returns:
        function that increases sigmoidally from 0 y_low to y_high
        the halfway point is at x_inflect on the x-axis and the slope
        at x_inflect goes to (0, y_low) if the multiplier is 1.
    """

    amplitude = y_high - y_low
    x_delta = x_inflect - x_start
    slope_at_inflection = multiplier * 0.5 * amplitude / x_delta
    b = 4. * slope_at_inflection / amplitude

    def curve(x):
        arg = b * ( x_inflect - x )
        # check for large values that will blow out exp
        if arg > 10.0:
            return y_low
        return amplitude / ( 1. + exp( arg ) ) + y_low

    return curve


def make_two_step_curve(
    y_low, y_med, y_high, x_start, x_med, x_end):

    curve1 = make_sigmoidal_curve(
        y_high=y_med, y_low=y_low, 
        x_start=x_start, x_inflect=(x_med-x_start)*0.5 + x_start, 
        multiplier=4)

    curve2 = make_sigmoidal_curve(
        y_high=y_high, y_low=y_med, 
        x_start=x_med, x_inflect=(x_end-x_med)*0.5 + x_med, 
        multiplier=4)

    def curve(x):
        if x < x_start:
            return y_low
        if x < x_med:
            return curve1(x)
        if x < x_end:
            return curve2(x)
        return y_high

    return curve


if __name__ == "__main__":

    import pylab
    import numpy
    #
    # x_vals = numpy.linspace(1950, 2050, 150)
    # curve = make_sigmoidal_curve(y_high=2, y_low=0, x_start=1950, x_inflect=1970)
    # pylab.plot(x_vals, map(curve, x_vals))
    # pylab.xlim(1950, 2020)
    # pylab.ylim(0, 5)
    # pylab.show()

    y_high = 3
    y_med = 0.5 * y_high
    curve = make_two_step_curve(0, y_med, y_high, 1950, 1995, 2015)
    x_vals = numpy.linspace(1950, 2050, 150)
    pylab.plot(x_vals, map(curve, x_vals))
    pylab.xlim(1950, 2020)
    pylab.ylim(0, 5)
    pylab.show()

