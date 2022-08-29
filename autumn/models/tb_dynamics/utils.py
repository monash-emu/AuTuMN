from scipy.integrate import quad
from copy import copy


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


def order_dict_by_keys(input_dict):
    """
    sort the input dictionary keys and return two separate lists with keys and values as lists with corresponding
        elements

    :param input_dict: dict
        dictionary to be sorted
    :return:
        :dict_keys: list
            sorted list of what were the dictionary keys
        : list
            values applicable to the sorted list of dictionary keys
    """
    dict_keys = list(input_dict.keys())
    dict_keys.sort()
    return dict_keys, [input_dict[key] for key in dict_keys]


def create_step_function_from_dict(input_dict):
    """
    create a step function out of dictionary with numeric keys and values, where the keys determine the values of the
        independent variable at which the steps between the output values occur

    :param input_dict: dict
        dictionary in standard format with numeric keys for the points at which the steps occur and numeric values for
            the values to be returned from these points onwards
    :return: function
        the function constructed from input data
    """
    dict_keys, dict_values = order_dict_by_keys(input_dict)

    def step_function(input_value):
        if input_value >= dict_keys[-1]:
            return dict_values[-1]
        else:
            for key in range(len(dict_keys)):
                if input_value < dict_keys[key + 1]:
                    return dict_values[key]

    return step_function


def get_parameter_dict_from_function(input_function, breakpoints, upper_value=100.0):
    """
    create a dictionary of parameter values from a continuous function, an arbitrary upper value and some breakpoints
        within which to evaluate the function
    """
    revised_breakpoints = copy(add_zero_to_age_breakpoints(breakpoints))
    revised_breakpoints.append(upper_value)
    param_values = []
    for n_breakpoint in range(len(revised_breakpoints) - 1):
        param_values.append(
            get_average_value_of_function(
                input_function,
                revised_breakpoints[n_breakpoint],
                revised_breakpoints[n_breakpoint + 1],
            )
        )
    return {str(key): value for key, value in zip(revised_breakpoints, param_values)}


def add_zero_to_age_breakpoints(breakpoints):
    """
    append a zero on to a list if there isn't one already present, for the purposes of age stratification

    :param breakpoints: list
        integers for the age breakpoints requested
    :return: list
        age breakpoints with the zero value included
    """
    return [0] + breakpoints if 0 not in breakpoints else breakpoints


def get_average_value_of_function(input_function, start_value, end_value):
    """
    use numeric integration to find the average value of a function between two extremes

    :param input_function: function
        function to be interrogated
    :param start_value: float
        lower limit of the independent variable over which to integrate the function
    :param end_value: float
        upper limit of the independent variable over which to integrate the function
    """
    return quad(input_function, start_value, end_value)[0] / (end_value - start_value)
