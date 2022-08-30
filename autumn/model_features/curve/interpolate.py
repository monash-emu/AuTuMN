"""Builder functions for interpolating data - replaces scale_up.py

build_static_sigmoidal_multicurve is the equivalent of scale_up_function(method=4),
with build_sigmoidal_multicurve producing a dynamic (parameterizable) version of this

"""

from computegraph import jaxify
from autumn.model_features.functional import binary_search_ge


def set_using_jax(use_jax):
    jaxify.set_using_jax(use_jax)
    modules = jaxify.get_modules()
    global fnp
    fnp = modules["numpy"]


# Initialize the module based on current jaxify status, and use the appropriate numpy
set_using_jax(jaxify.get_using_jax())


def _uncorrected_sigmoid(x: float, curvature: float) -> float:
    """Return a sigmoid of x (assumed to be between 0.0 and 1.0),
    whose shape and range depends on curvature

    Args:
        x: Input to the function
        curvature: Degree of sigmoidal flattening - 1.0 is linear, higher values increase smoothing

    Returns:
        Output of sigmoidal curve function
    """
    arg = curvature * (0.5 - x)
    return 1.0 / (1.0 + fnp.exp(arg))


def make_norm_sigmoid(curvature: float) -> callable:
    """
    Build a sigmoid function with fixed curvature, whose output is normalized to (0.0,1.0) across
    the (0.0,1.0) input range
    Args:
        curvature: See _uncorrected_sigmoid

    Returns:
        The normalized sigmoid function
    """
    offset = _uncorrected_sigmoid(0.0, curvature)
    scale = 1.0 / (1.0 - (offset * 2.0))

    def sig(x):
        return (_uncorrected_sigmoid(x, curvature) - offset) * scale

    return sig


def build_sigmoidal_multicurve(x_points: jaxify.Array, curvature=16.0) -> callable:
    """Build a sigmoidal smoothing function across points specified by
    x_points; the returned function takes the y values as arguments in
    form of a scale_data dict with keys [min, max, values, ranges]

    Args:
        x_points: x values to interpolate across
        curvature: Sigmoidal curvature.  Default produces the same behaviour as the old
                   scale_up_function

    Returns:
        The multisigmoidal function of (x, scale_data)
    """
    xranges = fnp.diff(fnp.array(x_points))
    x_points = fnp.array(x_points)

    xmin = x_points.min()
    xmax = x_points.max()
    xbounds = fnp.array([xmin, xmax])

    sig = make_norm_sigmoid(curvature)

    def get_curve_at_t(t, values, ranges):
        # idx = sum(t >= x_points) - 1
        idx = binary_search_ge(t, x_points)

        offset = t - x_points[idx]
        relx = offset / xranges[idx]
        rely = sig(relx)
        return values[idx] + (rely * ranges[idx])

    if jaxify.get_using_jax():
        from jax import lax

        def scaled_curve(t: float, ydata: dict):
            # Branch on whether t is in bounds
            bounds_state = sum(t > xbounds)
            branches = [
                lambda _, __, ___: ydata["min"],
                get_curve_at_t,
                lambda _, __, ___: ydata["max"],
            ]
            return lax.switch(bounds_state, branches, t, ydata["values"], ydata["ranges"])

    else:

        def scaled_curve(t: float, ydata: dict):
            if t < xmin:
                return ydata["min"]
            elif t >= xmax:
                return ydata["max"]

            return get_curve_at_t(t, ydata["values"], ydata["ranges"])

    return scaled_curve


def build_static_sigmoidal_multicurve(
    x_points: jaxify.Array, y_points: jaxify.Array, curvature=16.0
) -> callable:
    """Return a wrapped build_sigmoidal_multicurve function that operates against
    a static array of y_points

    Args:
        x_points: x points
        y_points: y points
        curvature: Sigmoidal curvature.  Default produces the same behaviour as the old
                   scale_up_function
    Returns:
        Fixed curve function
    """

    scale_data = get_scale_data(fnp.array(y_points))
    # scaled_curve = build_sigmoidal_multicurve(fnp.array(x_points), curvature)

    # def fixed_curve(t: float):
    #    return scaled_curve(t, scale_data)

    # return fixed_curve

    xranges = fnp.diff(fnp.array(x_points))
    x_points = fnp.array(x_points)

    xmin = x_points.min()
    xmax = x_points.max()
    xbounds = fnp.array([xmin, xmax])

    sig = make_norm_sigmoid(curvature)

    if jaxify.get_using_jax():
        from jax import lax

        ydata = scale_data

        def get_curve_at_t(t, values, ranges):
            # idx = sum(t >= x_points) - 1
            idx = binary_search_ge(t, x_points)  # - 1 ? FIXME: We _need_ to verify this!
            offset = t - x_points[idx]
            relx = offset / xranges[idx]
            rely = sig(relx)
            return values[idx] + (rely * ranges[idx])

        def scaled_curve(t: float):
            # Branch on whether t is in bounds
            bounds_state = sum(t > xbounds)
            branches = [
                lambda _, __, ___: ydata["min"],
                get_curve_at_t,
                lambda _, __, ___: ydata["max"],
            ]
            return lax.switch(bounds_state, branches, t, ydata["values"], ydata["ranges"])

    else:

        def _build_curve(x_val, xrange, ybase_val, yrange):
            def get_curve_at_t(t):
                offset = t - x_val
                relx = offset / xrange
                rely = sig(relx)
                return ybase_val + (rely * yrange)

            return get_curve_at_t

        curve_funcs = []
        for i, xrange in enumerate(xranges):
            x_val = x_points[i]
            ybase_val = scale_data["values"][i]
            yrange = scale_data["ranges"][i]
            get_curve_at_t = _build_curve(x_val, xrange, ybase_val, yrange)
            curve_funcs.append(get_curve_at_t)

        ymin = scale_data["min"]
        ymax = scale_data["max"]

        def scaled_curve(t: float):
            if t < xmin:
                return ymin
            elif t >= xmax:
                return ymax
            else:
                idx = sum(t >= x_points) - 1
                return curve_funcs[idx](t)

    return scaled_curve


def get_scale_data(points: jaxify.Array) -> dict:
    """
    Precompute min, max, and ranges for a set of data to be used in a scaling function such as that
    produced by build_sigmoidal_multicurve.  The onus is on the caller of this function to ensure
    they are the length expected by the target callee
    """
    ranges = fnp.diff(points)
    ymin = points[0]
    ymax = points[-1]

    data = {"min": ymin, "max": ymax, "values": points, "ranges": ranges}

    return data
