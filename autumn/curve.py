# -*- coding: utf-8 -*-

"""

Sigmoidal function to generate cost coverage curves and historiacl
input curves.

"""


from math import exp
import numpy


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


if __name__ == "__main__":

    curve1 = make_sigmoidal_curve(y_high=2, y_low=0, x_start=1950, x_inflect=1970, multiplier=4)
    curve2 = make_sigmoidal_curve(y_high=4, y_low=2, x_start=1990, x_inflect=2003, multiplier=4)
    curve = lambda x: curve1(x) if x < 1990 else curve2(x)

    import pylab
    x_vals = numpy.linspace(1950, 2050, 150)
    pylab.plot(x_vals, map(curve, x_vals))
    pylab.xlim(1950, 2020)
    pylab.ylim(0, 5)
    pylab.show()

