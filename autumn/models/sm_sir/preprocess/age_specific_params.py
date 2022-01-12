from autumn.tools.inputs import get_population_by_agegroup
import numpy as np


def convert_param_agegroups(source_parameters, iso3, region, age_groups):
    """
    Converts the source parameters to match the model age groups.
    :param source_parameters: a list of values provided by 5-year band, starting from 0-4
    """
    source_agebreaks = [str(5*i) for i in range(len(source_parameters))]
    total_pops_5year_bands = get_population_by_agegroup(source_agebreaks, iso3, region=region, year=2020)

    param_values = []
    for i, model_agegroup in enumerate(age_groups):
        age_indice_low = int(int(model_agegroup) / 5.)
        if i == len(age_groups) - 1:
            relevant_source_indices = list(range(age_indice_low, len(source_agebreaks)))
        else:
            age_indice_up = int(int(age_groups[i + 1]) / 5.) - 1
            relevant_source_indices = np.arange(age_indice_low, age_indice_up + 1, 1).tolist()

        model_agegroup_pop = 0
        param_val = 0.
        for relevant_source_index in relevant_source_indices:
            bin_pop = total_pops_5year_bands[relevant_source_index]
            model_agegroup_pop += bin_pop
            param_val += source_parameters[relevant_source_index] * bin_pop

        param_values.append(param_val / model_agegroup_pop)

    return param_values
