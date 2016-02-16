# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 18:07:45 2015

@author: ntdoan

Make cost coverage and cost outcome curve

TO DO LIST

1. MAKE IT PRODUCE GRAPHICAL CURVES
2. WHEN HAVING A SPENDING AMOUNT, USE THE COST-COVERAGE CURVE TO GET THE CORRESPONDING COVERAGE LEVEL
3. USE THE COVERAGE LEVEL VALUE FROM (2), USE THE COVERAGE-OUTCOME CURVE TO GET THE CORRESPONDING OUTCOME VALUE
4. USE THE OUTCOME VALUE FROM 3 AND FEED IT INTO JAMES' TRANSMISSION DYNAMIC MODEL
5. ADD MORE COMPLEXITY: MULTIPLE INTERVENTIONS, SAME INTERVENTIONS AFFECTING DIFFERENT OUTCOMES, COSTVERAGE LEVEL FOR EVERY GIVEN YEAR ETC.

"""


from math import exp
import numpy


def make_sigmoidal_fn(y_lo=0, y_hi=1.0, x_inf=0.5, multiplier=1.):
    """
    Args:
        y_lo: lowest y value
        y_hi: highest y value
        x_inf: inflection point of graph along the x-axis
        multiplier: if 1, slope at x_inf goes to (0, y_lo)

    Returns:
        function that increases sigmoidally from 0 y_lo to y_hi
        the halfway point is at x_inf on the x-axis and the slope
        at x_inf goes to (0, y_lo) if the multiplier is 1.
    """

    saturation = y_hi - y_lo
    slope_at_inf = multiplier * 0.5 * saturation / x_inf
    b = 4. * slope_at_inf / saturation
    def fn(x):
        arg = b * ( x_inf - x )
        if arg > 10.0:
            return y_lo
        return saturation / ( 1. + exp( arg ) ) + y_lo

    return fn


if __name__ == "__main__":
    import pylab
    x_big = 4000.
    fn = make_sigmoidal_fn(y_hi=.9, y_lo=.2, x_inf=x_big/2., multiplier=4)
    x_vals = numpy.linspace(0, x_big, 50)
    pylab.plot(x_vals, map(fn, x_vals))
    pylab.xlim(0, x_big)
    pylab.ylim(0, 1)
    pylab.show()
