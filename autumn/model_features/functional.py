from autumn.core import jaxify

fnp = jaxify.get_modules()["numpy"]

if jaxify.get_using_jax():
    # Jax only
    from jax import lax

    def piecewise_function(x, breakpoints, functions):
        index = sum(x >= breakpoints)
        return lax.switch(index, functions, x)

else:

    def piecewise_function(x, breakpoints, functions):
        index = sum(x >= breakpoints)
        return functions[index](x)


# All
def piecewise_constant(x, breakpoints, values):
    index = sum(x >= breakpoints)
    return values[index]


def windowed_constant(x: float, value: float, window_start: float, window_length: float):
    breakpoints = fnp.array((window_start, window_start + window_length))
    values = fnp.array((0.0, value, 0.0))
    return piecewise_constant(x, breakpoints, values)


def binary_search_ge(x: float, points: jaxify.Array) -> int:
    """Find the lowest index of the value within points
    to which x is greater than or equal to, using a binary
    search.

    A value of x lower than min(points) will return 0, and higher
    than max(points) will return len(points)-1

    Args:
        x: Value to find
        points: Array to search

    Returns:
        The index value satisying the above conditions
    """
    from jax import lax

    def cond(state):
        low, high = state
        return (high - low) > 1

    def body(state):
        low, high = state
        midpoint = (0.5 * (low + high)).astype(int)
        update_upper = x < points[midpoint]
        low = fnp.where(update_upper, low, midpoint)
        high = fnp.where(update_upper, midpoint, high)
        return (low, high)

    low, high = lax.while_loop(cond, body, (0, len(points) - 1))
    return lax.cond(x < points[high], lambda: low, lambda: high)
