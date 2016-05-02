# -*- coding: utf-8 -*-

"""

Sigmoidal and spline functions to generate cost coverage curves and historiacl
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
    if amplitude == 0:
        def curve(x):
            return y_low
        return curve

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

# The following function should no more be relevant as scale_up_function with argument method=4 is equivalent
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

def scale_up_function(x, y, method=3):
    """
    Given a set of points defined by x and y,
    this function fits a cubic spline and returns the interpolated function
    Input:
        x: The independent variable at each observed point
        y: The dependent variable at each observed point
        method: Select an interpolation method. Methods 1, 2 and 3 use cubic interpolation defined step by step.
            1: Uses derivative at each point as constrain --> see description of the encapsulated function
            derivatives(x, y). At each step, the defined portion of the curve covers the interval [x_i, x_(i+1)]
            2: Less constrained interpolation as the curve does not necessarily pass by every point.
            At each step, the defined portion of the curve covers the interval [x_i, x_(i+2)] except when the derivative
            should be 0 at x_(i+1). In this case, the covered interval is [x_i, x_(i+1)]
            3: The curve has to pass by every point. At each step, the defined portion of the curve covers
            the interval [x_i, x_(i+1)] but x_(i+2) is still used to obtain the fit on [x_i, x_(i+1)]
            4: Uses sigmoidal curves. This is a generalisation of the function make_two_step_curve
    Output:
        interpolation function
    """
    assert len(x) == len(y), "x and y must have the same length"

    if len(x) == 1:
        def curve(t):
            return y[0]
        return curve

    def derivatives(x, y):
        """
            Defines the slopes at each data point
            Should be 0 for the first and last points (x_0 and x_n)
            Should be 0 when the variation is changing (U or n shapes), i.e. when(y_i - y_(i-1))*(y_i - y_(i+1)) > 0

            If the variation is not changing, the slope is defined by a linear combination of the two slopes measured between
            the point and its two neighbours. This combination is weighted according to the distance with the neighbours.
            This is only relevant when method=1

        """

        v = numpy.zeros(len(x))  # initializes all zeros

        for i in range(len(x))[1:-1]:  # for each interior point
            if (y[i] - y[i - 1]) * (y[i] - y[i + 1]) < 0:  # not changing variation
                slope_left = (y[i] - y[i - 1]) / (x[i] - x[i - 1])
                slope_right = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
                w = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1])
                v[i] = w * slope_right + (1 - w) * slope_left
        return v

    x = numpy.array(x)
    y = numpy.array(y)

    if method in (1, 2, 3):
        # Normalise x to avoid too big or too small numbers when taking x**3
        coef = numpy.exp(numpy.log10(max(x)))
        x = x / coef

    vel = derivatives(x, y) # obtain derivatives conditions
    m = numpy.zeros((len(x) - 1, 4))  # to store the polynomial coefficients for each section [x_i, x_(i+1)]

    if method == 1:
        for i in range(1, len(x)):
            x0, x1 = x[i - 1], x[i]
            g = numpy.array([
                [x0 ** 3, x0 ** 2, x0, 1],
                [x1 ** 3, x1 ** 2, x1, 1],
                [3 * x0 ** 2, 2 * x0, 1, 0],
                [3 * x1 ** 2, 2 * x1, 1, 0],
            ])
            # bound conditions:  f(x0) = y0   f(x1) = y1  f'(x0) = v0  f'(x1) = v1
            d = numpy.array([y[i - 1], y[i], vel[i - 1], vel[i]])
            m[i - 1, :] = numpy.linalg.solve(g, d)
    elif method == 2:
        pass_next = 0
        for i in range(len(x))[0:-1]:
            if pass_next == 1:  # when = 1, the next section [x_(i+1), x_(i+2)] is already defined.
                pass_next = 0
            else:
                x0 = x[i]
                y0 = y[i]

                # define left velocity condition
                if vel[i] == 0:
                    v = 0
                else:
                    # get former polynomial to get left velocity condition. Derivative has to be continuous
                    p = m[i - 1, :]
                    v = 3 * p[0] * x0 ** 2 + 2 * p[1] * x0 + p[2]

                if vel[i + 1] == 0: # define only one section
                    x1 = x[i + 1]
                    y1 = y[i + 1]
                    g = numpy.array([
                        [x0 ** 3, x0 ** 2, x0, 1],
                        [x1 ** 3, x1 ** 2, x1, 1],
                        [3 * x0 ** 2, 2 * x0, 1, 0],
                        [3 * x1 ** 2, 2 * x1, 1, 0],

                    ])
                    # bound conditions: f(x0) = y0  f(x1) = y1   f'(x0) = v0  f'(x1) = 0
                    d = numpy.array([y0, y1, v, 0])
                    m[i, :] = numpy.linalg.solve(g, d)
                elif vel[i + 2] == 0: # defines two sections
                    x1, x2 = x[i + 1], x[i + 2]
                    y1, y2 = y[i + 1], y[i + 2]
                    g = numpy.array([
                        [x0 ** 3, x0 ** 2, x0, 1],
                        [x2 ** 3, x2 ** 2, x2, 1],
                        [3 * x0 ** 2, 2 * x0, 1, 0],
                        [3 * x2 ** 2, 2 * x2, 1, 0],

                    ])
                    # bound conditions: f(x0) = y0  f(x2) = y2   f'(x0) = v0  f'(x2) = 0
                    d = numpy.array([y0, y2, v, 0])
                    sol = numpy.linalg.solve(g, d)
                    m[i, :] = sol
                    m[i + 1, :] = sol
                    pass_next = 1
                else: # v1 and v2 are not null. We define two sections
                    x1, x2 = x[i + 1], x[i + 2]
                    y1, y2 = y[i + 1], y[i + 2]
                    g = numpy.array([
                        [x0 ** 3, x0 ** 2, x0, 1],
                        [x1 ** 3, x1 ** 2, x1, 1],
                        [x2 ** 3, x2 ** 2, x2, 1],
                        [3 * x0 ** 2, 2 * x0, 1, 0],
                    ])
                    # bound conditions: f(x0) = y0  f(x1) = y1 f(x2) = y2   f'(x0) = v0
                    d = numpy.array([y0, y1, y2, v])
                    sol = numpy.linalg.solve(g, d)
                    m[i, :] = sol
                    m[i + 1, :] = sol
                    pass_next = 1

    elif method == 3:
        pass_next = 0
        for i in range(len(x))[0:-1]:
            if pass_next == 1:  # when = 1, the next section [x_(i+1), x_(i+2)] is already defined.
                pass_next = 0
            else:
                x0 = x[i]
                y0 = y[i]

                # define left velocity condition
                if vel[i] == 0:
                    v = 0
                else:
                    # get former polynomial to get left velocity condition
                    p = m[i - 1, :]
                    v = 3 * p[0] * x0 ** 2 + 2 * p[1] * x0 + p[2]

                if vel[i + 1] == 0:
                    x1 = x[i + 1]
                    y1 = y[i + 1]
                    g = numpy.array([
                        [x0 ** 3, x0 ** 2, x0, 1],
                        [x1 ** 3, x1 ** 2, x1, 1],
                        [3 * x0 ** 2, 2 * x0, 1, 0],
                        [3 * x1 ** 2, 2 * x1, 1, 0],
                    ])
                    # bound conditions: f(x0) = y0  f(x1) = y1   f'(x0) = v0  f'(x1) = 0
                    d = numpy.array([y0, y1, v, 0])
                    m[i, :] = numpy.linalg.solve(g, d)
                else:
                    x1, x2 = x[i + 1], x[i + 2]
                    y1, y2 = y[i + 1], y[i + 2]
                    g = numpy.array([
                        [x0 ** 3, x0 ** 2, x0, 1],
                        [x1 ** 3, x1 ** 2, x1, 1],
                        [x2 ** 3, x2 ** 2, x2, 1],
                        [3 * x0 ** 2, 2 * x0, 1, 0],

                    ])
                    # bound conditions: f(x0) = y0  f(x1) = y1  f(x2) = y2  f'(x0) = v0
                    d = numpy.array([y0, y1, y2, v])
                    m[i, :] = numpy.linalg.solve(g, d)

    elif method == 4:
        functions = [[] for j in range(len(x))[0:-1]] # initializes an empty list to store functions
        for i in range(len(x))[0:-1]:
            y_high = y[i + 1]
            y_low = y[i]
            x_start = x[i]
            x_inflect = 0.5 * (x[i] + x[i+1])
            func = make_sigmoidal_curve(
                y_high=y[i + 1], y_low=y[i],
                x_start=x[i], x_inflect=0.5 * (x[i] + x[i+1]),
                multiplier=4)
            functions[i] = func

        def curve(t):
            if t <= min(x):  # t is before the range defined by x -> takes the initial value
                return y[0]
            elif t >= max(x):  # t is after the range defined by x -> takes the last value
                return y[-1]
            else:  # t is in the range defined by x
                index_low = len(x[x <= t]) - 1
                func = functions[index_low]
                return func(t)

        return curve

    else:
        print('method ' + method + 'does not exist.')

    def curve(t):
        t = t / coef
        if t <= x[0]:  # constant before x[0]
            return y[0]
        elif t >= x[-1]:  # constant after x[0]
            return y[-1]
        else:
            index = len(x[x <= t]) - 1
            p = m[index, :]  # corresponding coefficients
            y_t = p[0] * t ** 3 + p[1] * t ** 2 + p[2] * t + p[3]
            return y_t

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

    # y_high = 3
    # y_med = 0.5 * y_high
    # curve = make_two_step_curve(0, y_med, y_high, 1950, 1995, 2015)
    # x_vals = numpy.linspace(1950, 2050, 150)
    # pylab.plot(x_vals, map(curve, x_vals))
    # pylab.xlim(1950, 2020)
    # pylab.ylim(0, 5)
    # pylab.show()

    x = (1960, 1965, 1975, 1976, 1980, 1985, 1990, 1997, 2000, 2002, 2005)
    y = numpy.random.rand(len(x))

    f = scale_up_function(x, y, method=1)
    g = scale_up_function(x, y, method=2)
    h = scale_up_function(x, y, method=3)
    k = scale_up_function(x, y, method=4)

    x_vals = numpy.linspace(1950, 2010, 1000)
    pylab.plot(x_vals, map(f, x_vals), color='r')
    pylab.plot(x_vals, map(g, x_vals), color='b')
    pylab.plot(x_vals, map(h, x_vals), color='g')
    pylab.plot(x_vals, map(k, x_vals), color='purple')

    pylab.plot(x, y, 'ro')
    pylab.show()


