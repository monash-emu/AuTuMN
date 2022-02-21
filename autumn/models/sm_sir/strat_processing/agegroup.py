from typing import List, Dict, Union
import itertools

from autumn.tools.inputs import get_population_by_agegroup
from autumn.tools.utils.utils import weighted_average


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

    # Collate up the dictionary by modelled age groups
    relevant_source_indices = {}
    for i_age, model_agegroup in enumerate(modelled_age_groups):
        age_index_low = source_agebreaks.index(model_agegroup)
        if model_agegroup == modelled_age_groups[-1]:
            age_index_up = source_agebreaks[-1]
        else:
            age_index_up = source_agebreaks.index(modelled_age_groups[i_age + 1])
        relevant_source_indices[model_agegroup] = source_agebreaks[age_index_low: age_index_up]

    # Should be impossible for this check to fail
    msg = "Not all source age groups being mapped to modelled age groups"
    assert list(itertools.chain.from_iterable(relevant_source_indices.values())) == source_agebreaks, msg

    return relevant_source_indices


def convert_param_agegroups(
        iso3: str,
        region: Union[None, str],
        source_dict: Dict[int, float],
        modelled_age_groups: List[int],
) -> List[float]:
    """
    Converts the source parameters to match the model age groups.

    Args:
        iso3: Parameter for get_population_by_agegroup
        region: Parameter for get_population_by_agegroup
        source_dict: A list of parameter values provided according to 5-year band, starting from 0-4
        modelled_age_groups: Parameter for get_population_by_agegroup

    Returns:
        The list of the processed parameters in the format needed by the model

    """

    # Get default age brackets and the population structured with these default categories
    source_agebreaks = list(source_dict.keys())
    total_pops_5year_bands = get_population_by_agegroup(source_agebreaks, iso3, region=region, year=2020)
    total_pops_5year_dict = {age: pop for age, pop in zip(source_agebreaks, total_pops_5year_bands)}

    msg = "Modelled age group(s) incorrectly specified, not in standard age breaks"
    assert all([age_group in source_agebreaks for age_group in modelled_age_groups]), msg

    # Find out which of the standard source categories (values) apply to each modelled age group (keys)
    relevant_source_indices = get_relevant_indices(source_agebreaks, modelled_age_groups)

    # For each age bracket
    param_values = []
    for model_agegroup in modelled_age_groups:
        relevant_indices = relevant_source_indices[model_agegroup]
        weights = {k: total_pops_5year_dict[k] for k in relevant_indices}
        values = {k: source_dict[k] for k in relevant_indices}
        param_values.append(weighted_average(values, weights))

    return param_values
