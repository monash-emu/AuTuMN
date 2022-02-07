import numpy as np
from typing import List, Union

from autumn.tools.inputs import get_population_by_agegroup


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
        iso3:
        region:
        modelled_age_groups:

    Returns:
        The list of the processed parameters in the format needed by the model

    """

    # Get default age brackets and population structured in that way
    source_agebreaks = [str(5 * i) for i in range(len(source_parameters))]
    total_pops_5year_bands = get_population_by_agegroup(source_agebreaks, iso3, region=region, year=2020)

    print("________________________")
    # Calculate the parameter values for the modelled age groups
    param_values = []
    for i_age, model_agegroup in enumerate(modelled_age_groups):

        # If the last modelled age group
        if i_age == len(modelled_age_groups) - 1:
            relevant_source_indices = list(range(i_age, len(source_agebreaks)))

        else:
            relevant_source_indices = [i_age]

        model_agegroup_pop = 0
        param_val = 0.
        for relevant_source_index in relevant_source_indices:
            bin_pop = total_pops_5year_bands[relevant_source_index]
            model_agegroup_pop += bin_pop
            param_val += source_parameters[relevant_source_index] * bin_pop

        param_values.append(param_val / model_agegroup_pop)

    return param_values
