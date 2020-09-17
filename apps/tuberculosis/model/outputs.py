"""
FIXME: These all need tests.
"""
from typing import List, Dict
import numpy as np
from autumn.constants import Compartment


def make_infectious_prevalence_calculation_function(stratum_filters=[]):
    """
    Make a derived output function that calculates TB prevalence, stratified or non-stratified depending on the
    requested stratum_filters.
    :param stratum_filters: list of dictionaries.
    example of stratum_filters:
        [{'name': 'age', 'value': '15'}, {'name':'organ', 'value':'smear_positive'}]
    :return: a derived output function
    """
    def calculate_prevalence_infectious(time_idx, model, compartment_values, derived_outputs):
        """
        Calculate active TB prevalence at each time-step. Returns the proportion per 100,000 population.
        """
        prev_infectious = sum(
            [compartment_values[i] for i, compartment in enumerate(model.compartment_names) if
             (compartment.has_name(Compartment.INFECTIOUS) and all(
                 [compartment.has_stratum(stratum['name'], stratum['value']) for stratum in stratum_filters]
                )
              )
             ]
        )
        if len(stratum_filters) == 0:  # not necessary but will speed up calculation
            denominator = sum(compartment_values)
        else:
            denominator = calculate_stratum_population_size(compartment_values, model, stratum_filters)
        return 1.e5 * prev_infectious / denominator
    return calculate_prevalence_infectious


def make_latency_percentage_calculation_function(stratum_filters=[]):
    """
    Make a derived output function that calculates LTBI prevalence, stratified or non-stratified depending on the
    requested stratum_filters.
    :param stratum_filters: list of dictionaries.
    example of stratum_filters:
        [{'name': 'age', 'value': '15'}, {'name':'organ', 'value':'smear_positive'}]
    :return: a derived output function
    """
    def calculate_prevalence_infectious(time_idx, model, compartment_values, derived_outputs):
        """
        Calculate active TB prevalence at each time-step. Returns the proportion per 100,000 population.
        """
        prev_latent = sum(
            [compartment_values[i] for i, compartment in enumerate(model.compartment_names) if
             ((compartment.has_name(Compartment.EARLY_LATENT) or compartment.has_name(Compartment.EARLY_LATENT)) and
              all([compartment.has_stratum(stratum['name'], stratum['value']) for stratum in stratum_filters])
              )
             ]
        )
        if len(stratum_filters) == 0:  # not necessary but will speed up calculation
            denominator = sum(compartment_values)
        else:
            denominator = calculate_stratum_population_size(compartment_values, model, stratum_filters)
        return 100 * prev_latent / denominator
    return calculate_prevalence_infectious


def calculate_population_size(time_idx, model, compartment_values, derived_outputs):
    return sum(compartment_values)


def calculate_stratum_population_size(compartment_values, model, stratum_filters):
    """
    Calculate the population size of a given stratum.
    :param compartment_values: list of compartment sizes
    :param model: model object
    :param stratum_filters: list of dictionaries
    example of stratum_filters:
        [{'name': 'age', 'value': '15'}, {'name':'organ', 'value':'smear_positive'}]
    :return: float
    """
    relevant_filter_names = [s.name for s in model.stratifications if
                             set(s.compartments) == set(model.original_compartment_names)]

    # FIXME: we should be able to run this in a single line of code but the following comprehensive list does not work
    # return sum(
    #     [compartment_values[i] for i, compartment in enumerate(model.compartment_names) if
    #      all([compartment.has_stratum(stratum['name'], stratum['value']) for stratum in stratum_filters if
    #           stratum['name'] is relevant_filter_names])
    #      ]
    # )

    relevant_compartment_indices = []
    for i, compartment in enumerate(model.compartment_names):
        relevant = True
        for stratum in stratum_filters:
            if stratum['name'] in relevant_filter_names:
                if not compartment.has_stratum(stratum['name'], stratum['value']):
                   relevant = False
        if relevant:
            relevant_compartment_indices.append(i)
    return sum([compartment_values[i] for i in relevant_compartment_indices])


def get_all_derived_output_functions(calculated_outputs, outputs_stratification, model_stratifications):
    """
    Will automatically make and register the derived outputs based on the user-requested calculated_outputs and
    outputs_stratification
    :param calculated_outputs: list of requested derived outputs
    :param outputs_stratification: dict defining which derived outputs should be stratified and by which factor
    :param model_stratifications: dict containing all model stratifications
    :return: dict
    """
    simple_functions = {
        "population_size": calculate_population_size,
    }
    factory_functions = {
        "prevalence_infectious": make_infectious_prevalence_calculation_function,
        "percentage_latent": make_latency_percentage_calculation_function,
    }
    model_stratification_names = [s.name for s in model_stratifications]
    derived_output_functions = {}
    for requested_output in calculated_outputs:
        if requested_output in simple_functions:
            derived_output_functions[requested_output] = simple_functions[requested_output]
        elif requested_output in factory_functions:
            # add the unstratified output
            derived_output_functions[requested_output] = factory_functions[requested_output]()
            # add potential stratified outputs
            if requested_output in outputs_stratification:
                for stratification_name in outputs_stratification[requested_output]:
                    if stratification_name in model_stratification_names:
                        stratification = [model_stratifications[i] for i, s in enumerate(model_stratifications) if
                                          s.name == stratification_name][0]
                        for stratum in stratification.strata:
                            derived_output_functions[requested_output + "X" + stratification_name + "_" + stratum] = \
                                factory_functions[requested_output]([
                                    {'name': stratification_name, 'value': stratum}
                                ])
        else:
            raise ValueError(requested_output + " not among simple_functions or factory_functions")
    return derived_output_functions
