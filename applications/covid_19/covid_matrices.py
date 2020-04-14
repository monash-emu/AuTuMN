import numpy as np

from autumn.curve import scale_up_function


def build_covid_matrices(country, mixing_params):
    """
    Builds mixing matrices as functions of time for each scenario

    :param country: str
        Country for which the mixing matrices are needed
    :param mixing_params: dict
        Instructions for how the mixing matrices should vary with time, including for the baseline
    :return: dict
        Mixing matrices as a function of time collated together
    """
    from autumn.demography.social_mixing import load_all_prem_types

    # Note that this line of code would break for countries in the second half of the alphabet
    mixing_matrix_components = load_all_prem_types(country)

    def mixing_matrix_function(time):
        mixing_matrix = mixing_matrix_components['all_locations']
        for location in \
                [loc for loc in ['home', 'other_locations', 'school', 'work']
                 if loc + '_times' in mixing_params]:
            location_adjustment = \
                scale_up_function(
                    mixing_params[location + '_times'],
                    mixing_params[location + '_values']
                )
            mixing_matrix = \
                np.add(
                    mixing_matrix, (location_adjustment(time) + 1.) * mixing_matrix_components[location]
                )
        return mixing_matrix

    return mixing_matrix_function
