from typing import List, Dict, Union

import itertools

from autumn.tools.inputs import get_population_by_agegroup


def get_relevant_indices(
        source_agebreaks: List[int],
        modelled_age_groups: List[int],
) -> Dict[int, List[int]]:
    """
    Find the standard source age brackets relevant to each modelled age bracket.

    Args:
        source_agebreaks: The standard source age brackets (5-year brackets to 75+)
        modelled_age_groups: The age brackets being applied in the model

    Returns:
        The set of source age breaks applying to each modelled age group

    """

    relevant_source_indices = {}
    for i_age, model_agegroup in enumerate(modelled_age_groups):
        age_index_low = source_agebreaks.index(model_agegroup)
        if model_agegroup == modelled_age_groups[-1]:
            age_index_up = source_agebreaks[-1]
        else:
            age_index_up = source_agebreaks.index(modelled_age_groups[i_age + 1])
        relevant_source_indices[model_agegroup] = source_agebreaks[age_index_low: age_index_up]

    assert list(itertools.chain.from_iterable(relevant_source_indices.values())) == source_agebreaks

    return relevant_source_indices


def convert_param_agegroups(
        source_parameters: List[float],
        iso3: str,
        region: Union[None, str],
        modelled_age_groups: List[int],
) -> List[float]:
    """
    Converts the source parameters to match the model age groups.

    Args:
        source_parameters: A list of values provided by 5-year band, starting from 0-4
        iso3, region, modelled_age_groups: The parameters needed for obtaining the population

    Returns:
        The list of the processed parameters in the format needed by the model

    """

    # Get default age brackets and population structured with these default categories
    source_agebreaks = [5 * i for i in range(len(source_parameters))]
    total_pops_5year_bands = get_population_by_agegroup(source_agebreaks, iso3, region=region, year=2020)

    # Check modelled age brackets are specified in the expected way
    msg = "Modelled age brackets not multiples of 5, as expected"
    assert all([i_age % 5 == 0 for i_age in modelled_age_groups]), msg

    # Find out which of the standard source categories apply to each modelled age group
    relevant_source_indices = get_relevant_indices(source_agebreaks, modelled_age_groups)

    # Weight the parameter values according to the new structure
    param_values = []
    for i_age, model_agegroup in enumerate(modelled_age_groups):
        model_agegroup_pop = 0
        param_val = 0.
        for source_indices in relevant_source_indices[model_agegroup]:
            relevant_source_index = source_agebreaks.index(source_indices)
            bin_pop = total_pops_5year_bands[relevant_source_index]
            model_agegroup_pop += bin_pop
            param_val += source_parameters[relevant_source_index] * bin_pop
        param_values.append(param_val / model_agegroup_pop)

    return param_values
