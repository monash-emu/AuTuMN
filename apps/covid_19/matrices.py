import numpy as np
import matplotlib.pyplot as plt
from autumn.curve import scale_up_function
import copy


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
        # Make adjustments by location
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

        # Make adjustments by age
        affected_age_indices = [age_index for age_index in range(16) if
                               'age_' + str(age_index) + '_times' in mixing_params]
        complement_indices = [age_index for age_index in range(16) if age_index not in affected_age_indices]

        for age_index_affected in affected_age_indices:
            age_adjustment = scale_up_function(
                    mixing_params['age_' + str(age_index_affected) + "_times"],
                    mixing_params['age_' + str(age_index_affected) + "_values"],
                    method=4
                )
            for age_index_not_affected in complement_indices:
                mixing_matrix[age_index_affected, age_index_not_affected] *= age_adjustment(time)
                mixing_matrix[age_index_not_affected, age_index_affected] *= age_adjustment(time)

            # FIXME: patch for elderly cocooning in Victoria assuming 
            for age_index_affected_bis in affected_age_indices:
                mixing_matrix[age_index_affected, age_index_affected_bis] *= 1. - (1. - age_adjustment(time))/2.

        return mixing_matrix

    return mixing_matrix_function


def plot_mixing_params_over_time(mixing_params, npi_effectiveness_range):

    titles = {'home': 'Household', 'work': 'Workplace', 'school': 'School', 'other_locations': 'Other locations'}
    y_labs = {'home': 'h', 'work': 'w', 'school': 's', 'other_locations': 'l'}
    date_ticks = {61: '1/3', 76: '16/3', 92: '1/4', 107: '16/4', 122: '1/5', 137: '16/5', 152: '1/6'}
    # use italics for y_labs
    for key in y_labs:
        y_labs[key] = '$\it{' + y_labs[key] + '}$(t)'

    plt.style.use("ggplot")
    for i_loc, location in enumerate([
        loc
        for loc in ["home", "other_locations", "school", "work"]
        if loc + "_times" in mixing_params
    ]):
        plt.figure(i_loc)
        x = list(np.linspace(0.0, 152.0, num=10000))
        y = []
        for indice_npi_effect_range in [0, 1]:
            npi_effect = {key: val[indice_npi_effect_range] for key, val in npi_effectiveness_range.items()}

            modified_mixing_params = apply_npi_effectiveness(copy.deepcopy(mixing_params), npi_effect)

            location_adjustment = scale_up_function(
                modified_mixing_params[location + "_times"], modified_mixing_params[location + "_values"], method=4
            )

            _y = [location_adjustment(t) for t in x]
            y.append(_y)
            plt.plot(x, _y, color='navy')

        plt.fill_between(x, y[0], y[1], color='cornflowerblue')
        plt.xlim((60., 152.))
        plt.ylim((0, 1.1))

        plt.xticks(list(date_ticks.keys()), list(date_ticks.values()))
        plt.xlabel('Date in 2020')
        plt.ylabel(y_labs[location])
        plt.title(titles[location])
        plt.savefig('mixing_adjustment_' + location + '.png')


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


def update_mixing_parameters_for_prayers(mixing_params, t_start, participating_prop, other_location_multiplier,
                                         t_end=365):
    """
    Updare the mixing parameters to simulate re-installing regular Friday prayers from t_start to t_end. We assume that
    a proportion 'participating_prop' of the population participates in the prayers and that the other-location
    contact rates are multiplied by 'other_location_multiplier' for the participating individuals.
    """
    assert "other_locations_times" in mixing_params, "need to specify other_location mixing params"

    t = t_start
    reference_val = mixing_params['other_locations_values'][-1]
    amplified_val = reference_val * ((1. - participating_prop) + other_location_multiplier * participating_prop)
    while t < t_end:
        mixing_params['other_locations_times'] += [t, t+1, t+2]
        mixing_params['other_locations_values'] += [reference_val, amplified_val, reference_val]
        t += 7

    return mixing_params

