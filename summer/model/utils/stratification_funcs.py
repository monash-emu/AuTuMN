"""
Functions of functions for use in stratified models
"""


def create_multiplicative_function(multiplier):
    """
    return multiplication by a fixed value as a function

    :param multiplier: float
        value that the returned function multiplies by
    :return: function
        function that can multiply by the multiplier parameter when called
    """
    return lambda input_value, time: multiplier * input_value


def create_time_variant_multiplicative_function(time_variant_function):
    """
    similar to create_multiplicative_function, except that the value to multiply by can be a function of time, rather
        than a single value

    :param time_variant_function: function
        a function with the independent variable of time that returns the value that the input should be multiplied by
    :return: function
        function that will multiply the input value by the output value of the time_variant_function
    """
    return lambda input_value, time: time_variant_function(time) * input_value


def create_additive_function(increment):
    """
    return the addition of a fixed value as a function

    :param increment: float
        value that the returned function increments by
    :return: function
        function that can increment by the value parameter when called
    """
    return lambda value: value + increment


def create_sloping_step_function(start_x, start_y, end_x, end_y):
    """
    create sloping step function, returning start y-value for input values below starting x-value, ending y-value
        for input values above ending x-value and connecting slope through the middle

    :param start_x: float
        starting x-value
    :param start_y: float
        starting y-value
    :param end_x: float
        ending x-value
    :param end_y: float
        ending y-value
    :return: function
        sloping function as described above
    """
    gradient = (end_y - start_y) / (end_x - start_x)

    def step_function(age):
        if age < start_x:
            return start_y
        elif start_x <= age < end_x:
            return gradient * age + start_y - gradient * start_x
        elif end_x <= age:
            return end_y

    return step_function


def create_function_of_function(outer_function, inner_function):
    """
    function that can itself return a function that sequentially apply two functions and so can be used recursively to
        create a series of functions

    :param outer_function: function
        last function to be called
    :param inner_function: function
        first function to be called
    :return: function
        composite function that applies the inner and then the outer function, allowing the time parameter to be passed
            through if necessary
    """
    return lambda time: outer_function(inner_function(time), time)
