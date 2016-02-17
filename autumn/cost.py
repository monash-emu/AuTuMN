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


def make_sigmoidal_fn(y_low=0, y_high=1.0, x_inflect=0.5, multiplier=1.):
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
    slope_at_inflection = multiplier * 0.5 * amplitude / x_inflect
    b = 4. * slope_at_inflection / amplitude

    def fn(x):
        arg = b * ( x_inflect - x )
        # check for large values that will blow out exp
        if arg > 10.0:
            return y_low
        return amplitude / ( 1. + exp( arg ) ) + y_low

    return fn


if __name__ == "__main__":
    import pylab
    x_big = 4000.
    fn = make_sigmoidal_fn(y_high=.9, y_low=.2, x_inflect=0.5*x_big, multiplier=4)
    x_vals = numpy.linspace(0, x_big, 50)
    pylab.plot(x_vals, map(fn, x_vals))
    pylab.xlim(0, x_big)
    pylab.ylim(0, 1)
    pylab.show()
