import os
import pandas as pd
from copy import copy
import numpy as np

from autumn.settings.folders import INPUT_DATA_PATH
from autumn.core.inputs.social_mixing.constants import LOCATIONS
from autumn.core.inputs import get_population_by_agegroup
from autumn.core.inputs.demography.queries import _check_age_breakpoints

SOURCE_MATRICES_PATH = os.path.join(INPUT_DATA_PATH, "social-mixing", "socialmixr_outputs")

# Year when the contact survey was conducted (nearest multiple of 5)
REFERENCE_YEAR = {
    "VNM": 2005,  # survey conducted in 2007 in semi-rural setting
    "HKG": 2015,  # 2015-2016
    "CHN": 2015,  # 2017-2018 in Shanghai City
    # POLYMOD surveys below
    "GBR": 2005,  # survey conducted in 2005, 2006
    "ITA": 2005,
    "DEU": 2005,
    "FIN": 2005,
    "POL": 2005,
    "BEL": 2005,
    "NLD": 2005,
    "LUX": 2005,
    "FJI": 2015,
}


def build_synthetic_matrices(
    modelled_country_iso3,
    proxy_country_iso3,
    modelled_age_breaks,
    age_adjust,
    modelled_region_name=None,
    requested_locations=LOCATIONS,
):
    """
    :param modelled_country_iso3: The name of the modelled country
    :param proxy_country_iso3: The name of the country from which we want to source the contact matrix
    :param modelled_age_breaks: Lower bounds of the modelled age brackets
    :param age_adjust: Whether to apply adjustment based on population age distribution
    :param modelled_region_name: Name of sub-region if applicable
    :param requested_locations: List of locations for which a matrix should be built
    :return: dictionary containing the location-specific contact matrices for the modelled country
    """

    # Load contact matrices for the source country
    source_matrices, source_age_breaks = load_socialmixr_matrices(
        proxy_country_iso3, requested_locations
    )

    # adjust matrices for modelled region's age distribution
    if age_adjust:
        age_adjusted_matrices = adjust_matrices_for_age_distribution(
            source_matrices,
            proxy_country_iso3,
            modelled_country_iso3,
            source_age_breaks,
            modelled_region_name,
            requested_locations,
        )
    else:
        age_adjusted_matrices = source_matrices

    # convert matrices to match modelled age groups
    model_ready_matrices = convert_matrices_agegroups(
        age_adjusted_matrices,
        source_age_breaks,
        modelled_age_breaks,
        modelled_country_iso3,
        modelled_region_name,
        requested_locations,
    )

    return model_ready_matrices


def load_socialmixr_matrices(proxy_country_iso3, requested_locations):
    """
    Load location-specific matrices.
    Note that the matrices obtained from socialmixr use the convention that c_{i,j} is the average number of contacts
    aged j that an index of age i has.
    :param proxy_country_iso3: the ISO3 code of the country from which we read the socialmixer contact rates
    :param requested_locations: List of requested locations
    :return: location-specific contact matrices (in a dictionary) and list of associated lower age bounds
    """
    available_source_countries = os.listdir(SOURCE_MATRICES_PATH)
    assert (
        proxy_country_iso3 in available_source_countries
    ), f"No socialmixr data found for {proxy_country_iso3}"

    matrices = {}
    for i, location in enumerate(requested_locations):
        matrix_path = os.path.join(SOURCE_MATRICES_PATH, proxy_country_iso3, f"{location}.csv")
        msg = f"Could not find the required file {matrix_path}"
        assert f"{location}.csv" in os.listdir(
            os.path.join(SOURCE_MATRICES_PATH, proxy_country_iso3)
        ), msg

        loc_matrix = pd.read_csv(
            matrix_path,
        )
        # remove first column containing age brackets
        loc_matrix.drop(columns=loc_matrix.columns[0], axis=1, inplace=True)
        matrices[location] = loc_matrix.to_numpy()

        col_names = list(loc_matrix.columns)
        if i == 0:
            # work out age breakpoints
            ref_col_names = copy(col_names)
            age_breaks = [
                age_group.split(",")[0].split("+")[0].replace("[", "") for age_group in col_names
            ]
        else:
            # check that all matrices use the same age brackets in the same order
            assert col_names == ref_col_names

        # basic checks on age breaks
        _check_age_breakpoints(age_breaks)

    return matrices, age_breaks


def adjust_matrices_for_age_distribution(
    source_matrices,
    proxy_country_iso3,
    modelled_country_iso3,
    source_age_breaks,
    modelled_region_name=None,
    requested_locations=LOCATIONS,
):
    """
    Converts matrix based on the age distribution of the proxy country and that of the modelled country
    :param source_matrices: contact matrices for the proxy country
    :param proxy_country_iso3: the ISO3 code of the country from which we read the socialmixer contact rates
    :param modelled_country_iso3: the ISO3 code of the modelled country
    :param source_age_breaks: age breaks used in the source matrices
    :param modelled_region_name: name of a sub-region (if applicable)
    :param requested_locations: List of requested locations
    :return: contact matrices adjusted for the modelled country's age distribution (dictionary)
    """

    # Load age distributions of proxy country and modelled country
    age_proportions_proxy = _get_pop_props_by_age(
        source_age_breaks, proxy_country_iso3, None, REFERENCE_YEAR[proxy_country_iso3]
    )
    age_proportions_modelled = _get_pop_props_by_age(
        source_age_breaks, modelled_country_iso3, modelled_region_name, 2020
    )

    # Calculate age-specific population ratios
    assert all(
        [p > 0.0 for p in age_proportions_proxy]
    ), "All age proportions must be >0 to prevent division by zero."
    age_pop_ratio = [
        p_modelled / p_proxy
        for (p_modelled, p_proxy) in zip(age_proportions_modelled, age_proportions_proxy)
    ]

    # Convert into a diagonal matrix to prepare columns multiplication
    diag_age_pop_ratio = np.diag(age_pop_ratio)

    # Make population adjustment by multiplying matrices' columns by age-specific population ratios
    age_adjusted_matrices = {}
    for location in requested_locations:
        age_adjusted_matrices[location] = np.dot(source_matrices[location], diag_age_pop_ratio)

    return age_adjusted_matrices


def find_source_age_group_contributions(
    source_breaks, modelled_breaks, modelled_iso3, modelled_region
):

    # Get upper bounds for both classifications (assumed final band's upper bound = 100)
    modelled_upper_bounds = _get_upper_bounds(modelled_breaks)
    source_upper_bounds = _get_upper_bounds(source_breaks)

    # For each model's age group, work out the overlapping age portion with each source age group, and calculate
    # the proportion of the population this age portion takes up among the source age group.
    contributions_array = np.zeros((len(modelled_breaks), len(source_breaks)))

    for i_break, modelled_age_break in enumerate(modelled_breaks):
        model_lower, model_upper = int(modelled_age_break), modelled_upper_bounds[i_break]

        # Stores the portion of each source bracket included in a given modelled bracket
        for j_source, source_age_break in enumerate(source_breaks):

            # Work out the proportion of source bracket that is included in each modelled bracket
            source_lower, source_upper = int(source_age_break), source_upper_bounds[j_source]
            if model_upper > source_lower and model_lower < source_upper:
                overlap_range = max(source_lower, model_lower), min(source_upper, model_upper)
                contributions_array[
                    i_break, j_source
                ] = _get_proportion_between_ages_among_agegroup(
                    overlap_range, (source_lower, source_upper), modelled_iso3, modelled_region
                )

    _check_population_contributions_array(contributions_array)

    return contributions_array


def convert_matrices_agegroups(
    matrices,
    source_age_breaks,
    modelled_age_breaks,
    modelled_country_iso3,
    modelled_region_name=None,
    requested_locations=LOCATIONS,
):
    """
    Transform the contact matrices to match the model age stratification.
    :param matrices: contact matrices based the source age stratification
    :param source_age_breaks: age breaks of the source matrices
    :param modelled_age_breaks: the requested model's age breaks
    :param modelled_country_iso3: the ISO3 code of the modelled country
    :param modelled_region_name: name of a sub-region (if applicable)
    :param requested_locations: List of requested locations
    :return: contact matrices based on model's age stratification (dictionary)
    """
    source_age_break_contributions = find_source_age_group_contributions(
        source_age_breaks, modelled_age_breaks, modelled_country_iso3, modelled_region_name
    )

    n_modelled_groups = len(modelled_age_breaks)
    n_source_groups = len(source_age_breaks)

    # Build the output contact matrices based on the calculated age group contributions
    model_ready_matrices = {}
    for location in requested_locations:
        base_matrix = matrices[location]
        output_matrix = np.zeros((n_modelled_groups, n_modelled_groups))
        for i_model in range(n_modelled_groups):
            i_contributions = source_age_break_contributions[i_model, :].reshape(n_source_groups, 1)
            for j_model in range(n_modelled_groups):
                j_contributions = source_age_break_contributions[j_model, :].reshape(
                    n_source_groups, 1
                )

                # Sum over contactees' contributions for each contactor's contribution
                sums_over_contactees = np.dot(base_matrix, j_contributions)

                # Average over contactors' contributions
                average_over_contactors = float(
                    np.dot(np.transpose(i_contributions), sums_over_contactees)
                ) / float(sum(i_contributions))

                output_matrix[i_model, j_model] = average_over_contactors

        model_ready_matrices[location] = output_matrix

    if requested_locations == LOCATIONS:
        _check_model_ready_matrices(model_ready_matrices)
        model_ready_matrices = _clean_up_model_ready_matrices(model_ready_matrices)

    return model_ready_matrices


def _get_pop_props_by_age(age_breaks, country_iso3, region_name=None, reference_year=2020):
    age_pops = get_population_by_agegroup(
        age_breaks, country_iso3, region_name, year=reference_year
    )
    total_pop = sum(age_pops)
    return [p / total_pop for p in age_pops]


def _get_proportion_between_ages_among_agegroup(
    age_range_numerator, age_range_denominator, modelled_country_iso3, modelled_region_name
):
    """
    Work out the proportion of population aged within age_range_numerator among the age group age_range_denominator
    """
    if age_range_numerator == age_range_denominator:
        return 1.0

    numerator_low, numerator_up = age_range_numerator
    denominator_low, denominator_up = age_range_denominator

    assert numerator_low >= denominator_low and numerator_up <= denominator_up

    # get_population_by_agegroup has a strange signature, in that it insists on returning the bottom age group
    popsize_denominator = get_population_by_agegroup(
        [0, denominator_low, denominator_up], modelled_country_iso3, modelled_region_name, year=2020
    )[1]
    popsize_numerator = get_population_by_agegroup(
        [0, numerator_low, numerator_up], modelled_country_iso3, modelled_region_name, year=2020
    )[1]

    if numerator_low == denominator_low and numerator_up == denominator_up:
        result = 1.0
    elif popsize_denominator == 0:
        result = 0.0
    else:
        result = popsize_numerator / popsize_denominator

    assert 0.0 <= result <= 1.0
    return result


def _get_upper_bounds(all_age_breaks):
    """
    return the list of upper bounds associated with a given age stratification
    :param all_age_breaks: all lower bounds
    :return: list of integers. Final value assumed to be 100.
    """
    return [int(all_age_breaks[i + 1]) for i in range(len(all_age_breaks) - 1)] + [100]


def _check_model_ready_matrices(matrices):
    """
    Check that the all_locations matrix is approximately the sum of the 4 location-specific matrices
    """
    all_contacts = matrices["all_locations"]
    sum_contacts_by_location = (
        matrices["home"] + matrices["school"] + matrices["work"] + matrices["other_locations"]
    )
    error = all_contacts - sum_contacts_by_location
    print("Checking")
    assert (
        abs(error).max()
    ) < 1.0e-6, "The sum of the 4 location-specific matrices should match the all_locations matrix"


def _clean_up_model_ready_matrices(matrices):
    """
    Adjust the 'other locations' matrix such that the sum of the 4 location-specific matrices is equal to the
    all_locations matrix.
    """
    other_locations_matrix = matrices["all_locations"] - (
        matrices["home"] + matrices["school"] + matrices["work"]
    )
    other_locations_matrix[other_locations_matrix < 0.0] = 0
    matrices["other_locations"] = other_locations_matrix
    matrices["all_locations"] = (
        matrices["home"] + matrices["school"] + matrices["work"] + matrices["other_locations"]
    )
    return matrices


def _check_population_contributions_array(contributions_array):
    msg = "The sum of the contributions from each source age bin should be equal to 1"
    assert max(abs(np.sum(contributions_array, 0) - 1)) < 1.0e-3, msg
