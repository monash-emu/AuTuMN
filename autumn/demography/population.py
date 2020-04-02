from ..demography.ageing import add_agegroup_breaks
from autumn.db import find_population_by_agegroup


def get_population_size(model_parameters, input_database):
    """
    Calculate the population size by age-group, using UN data
    :param model_parameters: a dictionary containing model parameters
    :param input_database: database containing UN population data
    :return: a dictionary with the age-specific population sizes for the latest year available in UN data (2020)
    """
    if 'agegroup' in model_parameters['stratify_by']:
        model_parameters = add_agegroup_breaks(model_parameters)
        total_pops = find_population_by_agegroup(input_database,
                                                 [int(b) for b in model_parameters['all_stratifications']['agegroup']],
                                                 model_parameters['iso3'])[0]
    else:
        total_pops = find_population_by_agegroup(input_database, [0], model_parameters['iso3'])[0]

    total_pops = [int(1000. * total_pops[agebreak][-1]) for agebreak in list(total_pops.keys())]

    return total_pops, model_parameters
