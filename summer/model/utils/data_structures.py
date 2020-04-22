"""
Data manipulation functions
"""


def increment_list_by_index(list_to_increment, index_to_increment, increment_value):
    """
    very simple but general method to increment the odes by a specified value

    :param list_to_increment: list
        the list to be incremented, expected to be the list of ODEs
    :param index_to_increment: int
        the index of the list that needs to be incremented
    :param increment_value: float
        the value to increment the list by
    :return: list
        the list after incrementing
    """
    list_to_increment[index_to_increment] += increment_value
    return list_to_increment


def normalise_dict(value_dict):
    """
    normalise the values from a list using the total of all values, i.e. returning dictionary whose keys sum to one with
        same ratios between all the values in the dictionary after the function has been applied

    :param value_dict: dict
        dictionary whose values will be adjusted
    :return: dict
        same dictionary after values have been normalised to the total of the original values
    """
    return {key: value_dict[key] / sum(value_dict.values()) for key in value_dict}


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


def element_list_multiplication(list_1, list_2):
    """
    multiply elements of two lists to return another list with the same dimensions

    :param list_1: list
        first list of numeric values to be multiplied
    :param list_2: list
        second list of numeric values to be multiplied
    :return: list
        resulting list populated with the multiplied values
    """
    return [a * b for a, b in zip(list_1, list_2)]


def element_list_division(list_1, list_2):
    """
    divide elements of two lists to return another list with the same dimensions

    :param list_1: list
        first list of numeric values for numerators of division
    :param list_2: list
        second list of numeric values for denominators of division
    :return: list
        list populated with quotient values
    """
    return [a / b for a, b in zip(list_1, list_2)]


def convert_boolean_list_to_indices(list_of_booleans):
    """
    take a list of boolean values and return the indices of the elements containing True

    :param list_of_booleans: list
        sequence of booleans
    :return: list with integer values
        list of the values that were True in the input list_of_booleans
    """
    return [n_element for n_element, element in enumerate(list_of_booleans) if element]


def create_cumulative_dict(dict_of_props):

    cumulative_dict_of_props = {}
    cumulative_prop = 0.0
    for stratum in dict_of_props:
        cumulative_prop += dict_of_props[stratum]
        cumulative_dict_of_props[stratum] = cumulative_prop
    return cumulative_dict_of_props


def find_first_list_element_above(a_list, value):
    """
    Simple method to return the index of the first element of a list that is greater than a specified value.

    Args:
        a_list: List of floats
        value: The value that the element must be greater than
    """
    if max(a_list) <= value:
        ValueError("The requested value is greater than max(a_list)")

    for i, val in enumerate(a_list):
        if val > value:
            return i


def remove_multiple_elements_from_list(a_list, indices_to_be_removed):
    """
    remove list elements according to a list of indices to be removed from that list

    :param a_list: list
        list to be processed
    :param indices_to_be_removed: list
        list of the elements that are no longer needed
    """
    return [a_list[i] for i in range(len(a_list)) if i not in indices_to_be_removed]
