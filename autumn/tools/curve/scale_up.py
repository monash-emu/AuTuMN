"""
Sigmoidal and spline functions to generate cost coverage curves and historical input curves.
"""
from math import exp

import numpy as np
from scipy.interpolate import UnivariateSpline


def make_sigmoidal_curve(y_low=0, y_high=1.0, x_start=0, x_inflect=0.5, multiplier=1.0):

    """
    Base function to make sigmoidal curves. Not sure whether currently in use.
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
    b = 4.0 * slope_at_inflection / amplitude

    def curve(x):
        arg = b * (x_inflect - x)
        # check for large values that will blow out exp
        if arg > 10.0:
            return y_low

        return amplitude / (1.0 + exp(arg)) + y_low

    return curve


def make_linear_curve(x_0, x_1, y_0, y_1):
    assert x_1 > x_0
    slope = (y_1 - y_0) / (x_1 - x_0)

    def curve(x):
        return y_0 + slope * (x - x_0)

    return curve


""" the functions test_a and get_spare_fit are only used inside of scale_up_function when method=5 """


def test_a(a, x_min, x_max, x_peak, bound_low, bound_up):

    """
    Test function used to check that the cubic function defined by the parameter set 'a'
    will not overcome the bounds 'bound_low' and 'bound_up' on the interval [x_min, x_max]
    x_peak is one of x_min or x_max. At this point, the curve reaches an extremum equal to bound_low or bound_up
    Returns 0 when the curve overcomes one of the bounds
    """

    test = 1
    delta = 4.0 * a[2] ** 2 - 12.0 * a[1] * a[3]
    if delta > 0:
        zero_1 = (-2 * a[2] + delta ** 0.5) / (6.0 * a[3])
        zero_2 = (-2 * a[2] - delta ** 0.5) / (6.0 * a[3])
        zero = zero_1
        if abs(zero_2 - x_peak) > abs(zero_1 - x_peak):
            zero = zero_2
        y_zero = a[0] + a[1] * zero + a[2] * zero ** 2 + a[3] * zero ** 3

        if x_min < zero < x_max:
            if (y_zero > bound_up) or (y_zero < bound_low):
                test = 0
    return test


def get_spare_fit(
    indice, x_peak, bound, side, f, cut_off_dict, bound_low, bound_up, a_init, a_f, x, y
):

    """
    When a portion of the curve was detected outside of the bounds, we have tried to replace it by a portion that will
    reach an extremum on the bound that was overcome. However, in some cases, this new fit presents a singularity where
    it will still go over the bound and then change its variation to get back under the bound.
    In this case, we find a spare curve that will fit the spline to a point which is a bit further the nearest point.
    We use an iterative approach: We try to fit the curve to the nearest point. If the constraints are still not
    verified, we try the following point.
    In the worst case, this algorithm will stop by itself when reaching a point with a null gradient as this will
    represent a valid candidate.
    """

    ok = 0
    sg = 1.0
    if side == "left":
        spare_indice = indice
        sg = -1
    else:
        spare_indice = indice + 1
        sg = 1
    cpt = 0
    while ok == 0:
        cpt += 1
        spare_indice += sg
        x_spare = x[spare_indice]

        if spare_indice in cut_off_dict.keys():
            out = cut_off_dict[spare_indice]
            if x_spare < out["x_peak"]:
                a = out["a1"]
            else:
                a = out["a2"]
            y_spare = a[0] + a[1] * x_spare + a[2] * x_spare ** 2 + a[3] * x_spare ** 3
            v_spare = a[1] + 2.0 * a[2] * x_spare + 3.0 * a[3] * x_spare ** 2
        else:
            if spare_indice == 0:
                y_spare = y[0]
                v_spare = 0
            elif spare_indice == len(x) - 2:
                y_spare = a_f[0] + a_f[1] * x_spare + a_f[2] * x_spare ** 2 + a_f[3] * x_spare ** 3
                v_spare = a_f[1] + 2.0 * a_f[2] * x_spare + 3.0 * a_f[3] * x_spare ** 2
            elif spare_indice == len(x) - 1:
                y_spare = y[-1]
                v_spare = 0
            else:
                y_spare = f(x_spare)
                v_spare = f.derivatives(x_spare)[1]

        g = np.array(
            [
                [x_spare ** 3, x_spare ** 2, x_spare, 1],
                [x_peak ** 3, x_peak ** 2, x_peak, 1],
                [3.0 * x_spare ** 2, 2.0 * x_spare, 1, 0],
                [3.0 * x_peak ** 2, 2.0 * x_peak, 1, 0],
            ]
        )
        d = np.array([y_spare, bound, v_spare, 0.0])
        a = np.linalg.solve(g, d)
        a = a[::-1]  # reverse

        ok = test_a(a, min(x_spare, x_peak), max(x_spare, x_peak), x_peak, bound_low, bound_up)

    return {"a": a, "cpt": cpt}


def derivatives(x: np.ndarray, y: np.ndarray) -> np.ndarray:

    """
    Defines the slopes at each data point
    Should be 0 for the first and last points (x_0 and x_n)
    Should be 0 when the variation is changing (U or n shapes), i.e. when(y_i - y_(i-1))*(y_i - y_(i+1)) > 0
    If the variation is not changing, the slope is defined by a linear combination of the two slopes measured
    between the point and its two neighbours. This combination is weighted according to the distance with the
    neighbours.
    This is only relevant when method=1
    """
    v = np.zeros_like(x)  # Initialises all zeros
    for i in range(len(x))[1:-1]:  # For each interior point
        if (y[i] - y[i - 1]) * (y[i] - y[i + 1]) < 0:  # Not changing variation
            slope_left = (y[i] - y[i - 1]) / (x[i] - x[i - 1])
            slope_right = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
            w = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1])
            v[i] = w * slope_right + (1 - w) * slope_left
    return v


def scale_up_function(
    x,
    y,
    method=3,
    smoothness=1.0,
    bound_low=None,
    bound_up=None,
    auto_bound=1.3,
    intervention_end=None,
    intervention_start_date=None,
):
    """
    Given a set of points defined by x and y,
    this function fits a cubic spline and returns the interpolated function
    Args:
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
            5: Uses an adaptive algorithm producing either an interpolation or an approximation, depending on the value
            of the smoothness. This method allows for consideration of limits that are defined through bound_low and
            bound_up. See detailed description of this method in the code after the line "if method == 5:"
        smoothness, bound_up, bound_low, auto_bound are only used when method=5
        smoothness: Defines the level of smoothness of the curve. The minimum value is 0. and leads to an interpolation.
        bound_low: Defines a potential lower bound that the curve should not overcome.
        bound_up: Defines a potential upper bound that the curve should not overcome.
        auto_bound: In absence of bound_up or bound_low, sets a strip in which the curve should be contained.
                    Its value is a multiplier that applies to the amplitude of y to determine the width of the
                    strip. Set equal to None to delete this constraint.
        intervention_end: tuple or list of two elements defining the final time and the final level corresponding to a
                    potential intervention. If it is not None, an additional portion of curve will be added to define
                    the intervention through a sinusoidal function
        intervention_start_date: If not None, define the date at which intervention should start (must be >= max(x)).
                    If None, the maximal value of x will be used as a start date for the intervention. If the argument
                    'intervention_end' is not defined, this argument is not relevant and will not be used
    Returns:
        interpolation function
    """
    assert len(x) == len(y), "x and y must have the same length"
    if len(x) == 0:
        # Handle case where both times and values are empty.
        msg = "Cannot run scale_up_function on an empty sequence: x and y are empty lists."
        raise ValueError(msg)

    # Ensure that every element of x is unique.
    assert len(x) == len(set(x)), "There are duplicate values in x."

    x = np.array(x, dtype=np.float)
    y = np.array(y, dtype=np.float)

    # Ensure that the arrays are ordered
    order = x.argsort()
    x = x[order]
    y = y[order]

    # Define a scale-up for a potential intervention
    if intervention_end is not None:
        if intervention_start_date is not None:
            assert intervention_start_date >= max(
                x
            ), "The intervention start date should be >= max(x)"
            t_intervention_start = intervention_start_date
        else:
            t_intervention_start = max(x)
        curve_intervention = scale_up_function(
            x=[t_intervention_start, intervention_end[0]],
            y=[y[-1], intervention_end[1]],
            method=4,
        )

    if (len(x) == 1) or (max(y) - min(y) == 0):

        def curve(t, computed_values=None):
            if intervention_end is not None:
                if t >= t_intervention_start:
                    return curve_intervention(t)
                else:
                    return y[-1]
            else:
                return y[-1]

        return curve

    if method in (1, 2, 3):
        # Normalise x to avoid too big or too small numbers when taking x**3
        coef = np.exp(np.log10(max(x)))
        x = x / coef

    vel = derivatives(x, y)  # Obtain derivatives conditions
    m = np.zeros(
        (len(x) - 1, 4)
    )  # To store the polynomial coefficients for each section [x_i, x_(i+1)]

    if method == 1:
        for i in range(1, len(x)):
            x0, x1 = x[i - 1], x[i]
            g = np.array(
                [
                    [x0 ** 3, x0 ** 2, x0, 1],
                    [x1 ** 3, x1 ** 2, x1, 1],
                    [3 * x0 ** 2, 2 * x0, 1, 0],
                    [3 * x1 ** 2, 2 * x1, 1, 0],
                ]
            )
            # Bound conditions:  f(x0) = y0   f(x1) = y1  f'(x0) = v0  f'(x1) = v1
            d = np.array([y[i - 1], y[i], vel[i - 1], vel[i]])
            m[i - 1, :] = np.linalg.solve(g, d)
    elif method == 2:
        pass_next = 0
        for i in range(len(x))[0:-1]:
            if pass_next == 1:  # When = 1, the next section [x_(i+1), x_(i+2)] is already defined
                pass_next = 0
            else:
                x0 = x[i]
                y0 = y[i]

                # Define left velocity condition
                if vel[i] == 0:
                    v = 0
                else:
                    # Get former polynomial to get left velocity condition. Derivative has to be continuous
                    p = m[i - 1, :]
                    v = 3 * p[0] * x0 ** 2 + 2 * p[1] * x0 + p[2]

                if vel[i + 1] == 0:  # Define only one section
                    x1 = x[i + 1]
                    y1 = y[i + 1]
                    g = np.array(
                        [
                            [x0 ** 3, x0 ** 2, x0, 1],
                            [x1 ** 3, x1 ** 2, x1, 1],
                            [3 * x0 ** 2, 2 * x0, 1, 0],
                            [3 * x1 ** 2, 2 * x1, 1, 0],
                        ]
                    )
                    # Bound conditions: f(x0) = y0  f(x1) = y1   f'(x0) = v0  f'(x1) = 0
                    d = np.array([y0, y1, v, 0])
                    m[i, :] = np.linalg.solve(g, d)
                elif vel[i + 2] == 0:  # defines two sections
                    x1, x2 = x[i + 1], x[i + 2]
                    y1, y2 = y[i + 1], y[i + 2]
                    g = np.array(
                        [
                            [x0 ** 3, x0 ** 2, x0, 1],
                            [x2 ** 3, x2 ** 2, x2, 1],
                            [3 * x0 ** 2, 2 * x0, 1, 0],
                            [3 * x2 ** 2, 2 * x2, 1, 0],
                        ]
                    )
                    # Bound conditions: f(x0) = y0  f(x2) = y2   f'(x0) = v0  f'(x2) = 0
                    d = np.array([y0, y2, v, 0])
                    sol = np.linalg.solve(g, d)
                    m[i, :] = sol
                    m[i + 1, :] = sol
                    pass_next = 1
                else:  # v1 and v2 are not null. We define two sections
                    x1, x2 = x[i + 1], x[i + 2]
                    y1, y2 = y[i + 1], y[i + 2]
                    g = np.array(
                        [
                            [x0 ** 3, x0 ** 2, x0, 1],
                            [x1 ** 3, x1 ** 2, x1, 1],
                            [x2 ** 3, x2 ** 2, x2, 1],
                            [3 * x0 ** 2, 2 * x0, 1, 0],
                        ]
                    )
                    # Bound conditions: f(x0) = y0  f(x1) = y1 f(x2) = y2   f'(x0) = v0
                    d = np.array([y0, y1, y2, v])
                    sol = np.linalg.solve(g, d)
                    m[i, :] = sol
                    m[i + 1, :] = sol
                    pass_next = 1

    elif method == 3:
        pass_next = 0
        for i in range(len(x))[0:-1]:
            if pass_next == 1:  # When = 1, the next section [x_(i+1), x_(i+2)] is already defined
                pass_next = 0
            else:
                x0 = x[i]
                y0 = y[i]

                # Define left velocity condition
                if vel[i] == 0:
                    v = 0
                else:
                    # Get former polynomial to get left velocity condition
                    p = m[i - 1, :]
                    v = 3 * p[0] * x0 ** 2 + 2 * p[1] * x0 + p[2]

                if vel[i + 1] == 0:
                    x1 = x[i + 1]
                    y1 = y[i + 1]
                    g = np.array(
                        [
                            [x0 ** 3, x0 ** 2, x0, 1],
                            [x1 ** 3, x1 ** 2, x1, 1],
                            [3 * x0 ** 2, 2 * x0, 1, 0],
                            [3 * x1 ** 2, 2 * x1, 1, 0],
                        ]
                    )

                    # Bound conditions: f(x0) = y0  f(x1) = y1   f'(x0) = v0  f'(x1) = 0
                    d = np.array([y0, y1, v, 0])
                    m[i, :] = np.linalg.solve(g, d)
                else:
                    x1, x2 = x[i + 1], x[i + 2]
                    y1, y2 = y[i + 1], y[i + 2]
                    g = np.array(
                        [
                            [x0 ** 3, x0 ** 2, x0, 1],
                            [x1 ** 3, x1 ** 2, x1, 1],
                            [x2 ** 3, x2 ** 2, x2, 1],
                            [3 * x0 ** 2, 2 * x0, 1, 0],
                        ]
                    )
                    # Bound conditions: f(x0) = y0  f(x1) = y1  f(x2) = y2  f'(x0) = v0
                    d = np.array([y0, y1, y2, v])
                    m[i, :] = np.linalg.solve(g, d)

    elif method == 4:
        functions = [
            [] for j in range(len(x))[0:-1]
        ]  # Initialises an empty list to store functions
        for i in range(len(x))[0:-1]:
            func = make_sigmoidal_curve(
                y_high=y[i + 1],
                y_low=y[i],
                x_start=x[i],
                x_inflect=0.5 * (x[i] + x[i + 1]),
                multiplier=4,
            )
            functions[i] = func

        x_min = x.min()
        x_max = x.max()

        def curve(t, computed_values=None):
            if t <= x_min:  # t is before the range defined by x -> takes the initial value
                return y[0]
            elif t >= x_max:  # t is after the range defined by x -> takes the last value
                if intervention_end is not None:
                    if t >= t_intervention_start:
                        return curve_intervention(t)
                    else:
                        return y[-1]
                else:
                    return y[-1]
            else:  # t is in the range defined by x
                index_low = len(x[x <= t]) - 1
                func = functions[index_low]
                return func(t)

        return curve

    elif method == 5:

        """
        This method produces an approximation (or interpolation when smoothness=0.0) using cubic splines. When the
        arguments 'bound_low' or 'bound_up' are not None, the approximation is constrained so that the curve does not
        overcome the bounds.
        A low smoothness value will provide a curve that passes close to every point but leads to more variation changes
        and a greater curve energy. Set this argument equal to 0.0 to obtain an interpolation.
        A high smoothness value will provide a very smooth curve but its distance to certain points may be large.
        We use the following approach:
        1. We create a curve (f) made of cubic spline portions that approximates/interpolates the data, without any
            constraints
        2. We modify the initial and final sections of the curve in order to have:
            a. Perfect hits at the extreme points
            b. Null gradients at the extreme points
        3. If bounds are defined, we detect potential sections where they are overcome and we use the following method
            to adjust the fit:
            3.1. We detect the narrowest interval [x_i, x_j] which contains the overcoming section
            3.2. We fit a cubic spline (g) verifying the following conditions:
                a. g(x_i) = f(x_i)  and  g(x_j) = f(x_j)
                b. g'(x_i) = f'(x_i)  and  g'(x_j) = f'(x_j)
            3.3. If g still presents irregularities, we repeat the same process from 3.1. on the interval [x_(i-1), x_j]
                or [x_i, x_(j+1)], depending on which side of the interval led to an issue
        """

        # Calculate an appropriate smoothness (adjusted by smoothness)
        rmserror = 0.05
        s = smoothness * len(x) * (rmserror * np.fabs(y).max()) ** 2

        # Get rid of first elements when they have same y (same for last elements) as there is no need for fitting
        ind_start = 0
        while y[ind_start] == y[ind_start + 1]:
            ind_start += 1

        ind_end = len(x) - 1
        while y[ind_end] == y[ind_end - 1]:
            ind_end -= 1

        x = x[ind_start : (ind_end + 1)]
        y = y[ind_start : (ind_end + 1)]

        k = min(3, len(x) - 1)

        w = np.ones(len(x))
        w[0] = 5.0
        w[-1] = 5.0

        f = UnivariateSpline(x, y, k=k, s=s, ext=3, w=w)  # Create a first raw approximation

        # Shape the initial and final parts of the curve in order to get null gradients and to hit the external points
        x0 = x[0]
        x1 = x[1]
        x_f = x[-1]
        x_a = x[-2]

        v1 = f.derivatives(x1)[1]  #
        v_a = f.derivatives(x_a)[1]

        g = np.array(
            [
                [x0 ** 3, x0 ** 2, x0, 1],
                [x1 ** 3, x1 ** 2, x1, 1],
                [3 * x0 ** 2, 2 * x0, 1, 0],
                [3 * x1 ** 2, 2 * x1, 1, 0],
            ]
        )
        d_init = np.array([y[0], f(x1), 0, v1])
        a_init = np.linalg.solve(g, d_init)
        a_init = a_init[::-1]  # Reverse

        h = np.array(
            [
                [x_a ** 3, x_a ** 2, x_a, 1],
                [x_f ** 3, x_f ** 2, x_f, 1],
                [3 * x_a ** 2, 2 * x_a, 1, 0],
                [3 * x_f ** 2, 2 * x_f, 1, 0],
            ]
        )
        d_f = np.array([f(x_a), y[-1], v_a, 0])
        a_f = np.linalg.solve(h, d_f)
        a_f = a_f[::-1]  # Reverse

        # We have to make sure that the obtained fits do not go over/under the bounds
        cut_off_dict = {}

        amplitude = auto_bound * (y.max() - y.min())
        if bound_low is None:
            if auto_bound is not None:
                bound_low = 0.5 * (y.max() + y.min()) - 0.5 * amplitude
        if bound_up is None:
            if auto_bound is not None:
                bound_up = 0.5 * (y.max() + y.min()) + 0.5 * amplitude

        if (bound_low is not None) or (bound_up is not None):
            # We adjust the data so that no values go over/under the bounds
            if bound_low is not None:
                for i in range(len(x) - 1):
                    if y[i] < bound_low:
                        y[i] = bound_low

            if bound_up is not None:
                for i in range(len(x)):
                    if y[i] > bound_up:
                        y[i] = bound_up

            # Check bounds
            def cut_off(index, bound_low, bound_up, sign):
                if sign == -1:
                    bound = bound_low
                else:
                    bound = bound_up

                x0 = x[index]
                if index == 0:
                    y0 = y[0]
                else:
                    y0 = f(x0)

                # Look for the next knot at which the spline is inside of the bounds
                go = 1
                k = 0
                while go == 1:
                    k += 1

                    if (index + k) == 0:
                        y_k = y[0]
                    elif (index + k) == len(x) - 2:
                        y_k = (
                            a_f[0]
                            + a_f[1] * x[index + k]
                            + a_f[2] * x[index + k] ** 2
                            + a_f[3] * x[index + k] ** 3
                        )
                    elif (index + k) == len(x) - 1:
                        y_k = y[-1]
                    else:
                        y_k = f(x[index + k])

                    if (y_k <= bound_up) and (y_k >= bound_low):
                        go = 0

                next_index = index + k

                x1 = x[next_index]
                y1 = f(x1)

                if y0 == y1:
                    x_peak = 0.5 * (x0 + x1)
                else:
                    if y0 == bound:
                        x_peak = x0
                    elif y1 == bound:
                        x_peak = x1
                    else:
                        # Weighted positioning of the contact with bound
                        x_peak = x0 + (abs(y0 - bound) / (abs(y0 - bound) + abs(y1 - bound))) * (
                            x1 - x0
                        )

                if index == 0:
                    v0 = 0
                else:
                    v0 = f.derivatives(x0)[1]

                if index == (len(x) - 2):
                    v1 = 0
                else:
                    v1 = f.derivatives(x1)[1]

                if x0 != x_peak:
                    g = np.array(
                        [
                            [x0 ** 3, x0 ** 2, x0, 1],
                            [x_peak ** 3, x_peak ** 2, x_peak, 1],
                            [3.0 * x0 ** 2, 2.0 * x0, 1, 0],
                            [3.0 * x_peak ** 2, 2.0 * x_peak, 1, 0],
                        ]
                    )
                    d = np.array([y0, bound, v0, 0.0])
                    a1 = np.linalg.solve(g, d)
                    a1 = a1[::-1]  # Reverse

                if x1 != x_peak:
                    g = np.array(
                        [
                            [x_peak ** 3, x_peak ** 2, x_peak, 1],
                            [x1 ** 3, x1 ** 2, x1, 1],
                            [3.0 * x_peak ** 2, 2.0 * x_peak, 1, 0],
                            [3.0 * x1 ** 2, 2.0 * x1, 1, 0],
                        ]
                    )
                    d = np.array([bound, y1, 0.0, v1])
                    a2 = np.linalg.solve(g, d)
                    a2 = a2[::-1]  # Reverse

                if x0 == x_peak:
                    a1 = a2
                if x1 == x_peak:
                    a2 = a1

                indice_first = index
                t1 = test_a(a1, x0, x_peak, x_peak, bound_low, bound_up)
                if t1 == 0:  # There is something wrong here
                    spare = get_spare_fit(
                        index,
                        x_peak,
                        bound,
                        "left",
                        f,
                        cut_off_dict,
                        bound_low,
                        bound_up,
                        a_init,
                        a_f,
                        x,
                        y,
                    )
                    a1 = spare["a"]
                    indice_first = index - spare["cpt"]
                t2 = test_a(a2, x_peak, x1, x_peak, bound_low, bound_up)
                if t2 == 0:  # There is something wrong here
                    spare = get_spare_fit(
                        index,
                        x_peak,
                        bound,
                        "right",
                        f,
                        cut_off_dict,
                        bound_low,
                        bound_up,
                        a_init,
                        a_f,
                        x,
                        y,
                    )
                    a2 = spare["a"]
                    next_index = index + 1 + spare["cpt"]

                out = {
                    "a1": a1,
                    "a2": a2,
                    "x_peak": x_peak,
                    "indice_first": indice_first,
                    "indice_next": next_index,
                }
                return out

            t = x[0]

            while t < x[-1]:
                ok = 1
                if t == x[0]:
                    y_t = y[0]
                elif t < x[1]:
                    y_t = a_init[0] + a_init[1] * t + a_init[2] * t ** 2 + a_init[3] * t ** 3
                elif t < x[-2]:
                    y_t = f(t)
                elif t == x[-1]:
                    y_t = y[-1]
                else:
                    y_t = a_f[0] + a_f[1] * t + a_f[2] * t ** 2 + a_f[3] * t ** 3

                if bound_low is not None:
                    if y_t < bound_low:
                        ok = 0
                        sign = -1.0
                if bound_up is not None:
                    if y_t > bound_up:
                        ok = 0
                        sign = 1.0

                if ok == 0:
                    indice = len(x[x < t]) - 1
                    out = cut_off(indice, bound_low, bound_up, sign)

                    for k in range(out["indice_first"], out["indice_next"]):
                        cut_off_dict[k] = out
                    t = x[out["indice_next"]]
                t += (x[-1] - x[0]) / 1000.0

        def curve(t, computed_values=None):
            t = float(t)
            y_t = 0
            if t <= x[0]:
                y_t = y[0]
            elif t > x[-1]:
                if intervention_end is not None:
                    if t >= t_intervention_start:
                        y_t = curve_intervention(t)
                    else:
                        y_t = y[-1]
                else:
                    y_t = y[-1]
            elif x[0] < t < x[1]:
                y_t = a_init[0] + a_init[1] * t + a_init[2] * t ** 2 + a_init[3] * t ** 3
            elif x[-2] < t < x[-1]:
                y_t = a_f[0] + a_f[1] * t + a_f[2] * t ** 2 + a_f[3] * t ** 3
            else:
                y_t = f(t)

            if x[0] < t < x[-1]:
                indice = len(x[x < t]) - 1
                if indice in cut_off_dict.keys():
                    out = cut_off_dict[indice]
                    if t < out["x_peak"]:
                        a = out["a1"]
                    else:
                        a = out["a2"]
                    y_t = a[0] + a[1] * t + a[2] * t ** 2 + a[3] * t ** 3

            if (bound_low is not None) and (t <= x[-1]):
                y_t = max((y_t, bound_low))  # Security check. Normally not needed
            if (bound_up is not None) and (t <= x[-1]):
                y_t = min((y_t, bound_up))  # Security check. Normally not needed

            return float(y_t)

        return curve

    else:
        raise Exception("method " + method + "does not exist.")

    def curve(t, computed_values=None):
        t = t / coef
        if t <= x[0]:  # Constant before x[0]
            return float(y[0])
        elif t >= x[-1]:  # Constant after x[0]
            if intervention_end is not None:
                if t >= t_intervention_start / coef:
                    return float(curve_intervention(t * coef))
                else:
                    return float(y[-1])
            else:
                return float(y[-1])
        else:
            index = len(x[x <= t]) - 1
            p = m[index, :]  # Corresponding coefficients
            y_t = p[0] * t ** 3 + p[1] * t ** 2 + p[2] * t + p[3]
            return float(y_t)

    return curve
