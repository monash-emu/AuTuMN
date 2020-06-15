"""
String manipulation functions
"""
from functools import lru_cache


def create_stratum_name(stratification_name, stratum_name, joining_string="X"):
    """
    generate a name string to represent a particular stratum within a requested stratification

    :param stratification_name: str
        the "stratification" or rationale for implementing the current stratification process
    :param stratum_name: str
        name of the stratum within the stratification
    :param joining_string: str
        the character to add to the front to indicate that this string is the extension of the existing one
        in SUMMER, capitals are reserved for non-user-requested strings, in this case "X" is used as the default
    :return: str
        the composite string for the stratification
    """
    return joining_string + "%s_%s" % (stratification_name, str(stratum_name))


def create_stratified_name(stem, stratification_name, stratum_name):
    """
    generate a standardised stratified compartment name

    :param stem: str
        the previous stem to the compartment or parameter name that needs to be extended
    :param stratification_name: str
        the "stratification" or rationale for implementing the current stratification process
    :param stratum_name: str
        name of the stratum within the stratification
    :return: str
        the composite name with the standardised stratification name added on to the old stem
    """
    return stem + create_stratum_name(stratification_name, stratum_name)


def extract_x_positions(parameter, joining_string="X"):
    """
    find the positions within a string which are X and return as list, including length of list

    :param parameter: str
        the string for interrogation
    :param joining_string: str
        the string of interest whose character positions need to be found
    :return: list
        list of all the indices for where the X character occurs in the string, along with the total length of the list
    """
    return [
        loc for loc in range(len(parameter)) if parameter[loc] == joining_string
    ] + [len(parameter)]


def extract_reversed_x_positions(parameter):
    """
    find the positions within a string which are X and return as list reversed, including length of list

    :params and return: see extract_x_positions above
    """
    result = extract_x_positions(parameter)
    result.reverse()
    return result


@lru_cache(maxsize=None)
def find_stem(stratified_string: str):
    """
    find the stem of the compartment name as the text leading up to the first occurrence of the joining string
    should run slightly faster than using find_name_components
    """
    return stratified_string.split("X")[0]


@lru_cache(maxsize=None)
def find_name_components(compartment):
    """
    extract all the components of a stratified compartment or parameter name, including the stem

    :param compartment: str
        name of the compartment or parameter to be interrogated
    :return: list
        the extracted compartment components
    """

    # add -1 at the start, which becomes zero to represent the start when one is added
    x_positions = [-1] + extract_x_positions(compartment)

    # add one to the first index to go past the joining character, but not at the end
    return [
        compartment[x_positions[x_pos] + 1 : x_positions[x_pos + 1]]
        for x_pos in range(len(x_positions) - 1)
    ]


def find_stratum_index_from_string(
    compartment, stratification, remove_stratification_name=True
):
    """
    finds the stratum which the compartment (or parameter) name falls in when provided with the compartment name and the
        name of the stratification of interest
    for example, if the compartment name was infectiousXhiv_positiveXdiabetes_none and the stratification of interest
        provided through the stratification argument was hiv, then return positive

    :param compartment: str
        name of the compartment (or parameter) to be interrogated
    :param stratification: str
        the stratification of interest
    :param remove_stratification_name: bool
        whether to remove the stratification name and its trailing _ from the string to return
    :return: str
        the name of the stratum within which the compartment falls
    """
    stratum_name = [
        name
        for n_name, name in enumerate(find_name_components(compartment))
        if stratification in name
    ][0]
    return (
        stratum_name[stratum_name.find("_") + 1 :]
        if remove_stratification_name
        else stratum_name
    )
