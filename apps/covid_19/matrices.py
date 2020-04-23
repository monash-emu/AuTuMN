import numpy as np
import matplotlib.pyplot as plt
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
    # plot_mixing_params_over_time(mixing_params)
    from autumn.demography.social_mixing import load_all_prem_types

    # Note that this line of code would break for countries in the second half of the alphabet
    mixing_matrix_components = load_all_prem_types(country)

    def mixing_matrix_function(time):
        mixing_matrix = mixing_matrix_components["all_locations"]
        for location in [
            loc
            for loc in ["home", "other_locations", "school", "work"]
            if loc + "_times" in mixing_params
        ]:
            location_adjustment = scale_up_function(
                mixing_params[location + "_times"], mixing_params[location + "_values"], method=4
            )
            mixing_matrix = np.add(
                mixing_matrix,
                (location_adjustment(time) - 1.0) * mixing_matrix_components[location],
            )
        return mixing_matrix

    return mixing_matrix_function


def plot_mixing_params_over_time(mixing_params):

    for location in [
        loc
        for loc in ["home", "other_locations", "school", "work"]
        if loc + "_times" in mixing_params
    ]:

        location_adjustment = scale_up_function(
            mixing_params[location + "_times"], mixing_params[location + "_values"], method=4
        )

        x = list(np.linspace(0.0, 120.0, num=1000))

        y = [location_adjustment(t) for t in x]
        plt.plot(x, y)
        plt.show()


def apply_npi_effectiveness(mixing_params, npi_effectiveness):
    """
    Adjust the mixing parameters according by scaling them according to NPI effectiveness
    :param mixing_params: dict
        Instructions for how the mixing matrices should vary with time, including for the baseline
    :param npi_effectiveness: dict
        Instructions for how the input mixing parameters should be adjusted to account for the level of
        NPI effectiveness. mixing_params are unchanged if all NPI effectiveness values are 1.
    :return: dict
        Adjusted instructions
    """
    for location in [
        loc
        for loc in ["home", "other_locations", "school", "work"]
        if loc + "_times" in mixing_params
    ]:
        if location in npi_effectiveness:
            mixing_params[location + '_values'] = [1. - (1. - val) * npi_effectiveness[location]
                                                   for val in mixing_params[location + '_values']]

    return mixing_params


