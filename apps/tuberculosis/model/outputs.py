"""
FIXME: These all need tests.
"""
from typing import List
from summer.constants import Compartment
from summer.model.derived_outputs import (
    InfectionDeathFlowOutput,
    TransitionFlowOutput,
)
import numpy as np
from autumn.curve import tanh_based_scaleup

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
            [
                compartment_values[i]
                for i, compartment in enumerate(model.compartment_names)
                if (
                    (
                        compartment.has_name(Compartment.INFECTIOUS)
                        or compartment.has_name(Compartment.ON_TREATMENT)
                    )
                    and all(
                        [
                            compartment.has_stratum(stratum["name"], stratum["value"])
                            for stratum in stratum_filters
                        ]
                    )
                )
            ]
        )
        if len(stratum_filters) == 0:  # not necessary but will speed up calculation
            denominator = sum(compartment_values)
        else:
            denominator = calculate_stratum_population_size(
                compartment_values, model, stratum_filters
            )
        return 1.0e5 * prev_infectious / denominator

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
            [
                compartment_values[i]
                for i, compartment in enumerate(model.compartment_names)
                if (
                    (
                        compartment.has_name(Compartment.EARLY_LATENT)
                        or compartment.has_name(Compartment.LATE_LATENT)
                    )
                    and all(
                        [
                            compartment.has_stratum(stratum["name"], stratum["value"])
                            for stratum in stratum_filters
                        ]
                    )
                )
            ]
        )
        if len(stratum_filters) == 0:  # not necessary but will speed up calculation
            denominator = sum(compartment_values)
        else:
            denominator = calculate_stratum_population_size(
                compartment_values, model, stratum_filters
            )
        return 100 * prev_latent / denominator

    return calculate_prevalence_infectious


def get_notifications_connections(comps: List[Compartment], source_stratum=None):
    """
    Track "notifications": flow from infectious to treatment compartment.
    """
    output_name = "notifications"
    if source_stratum:
        output_name += "X" + source_stratum[0] + "_" + source_stratum[1]
        source_strata = {source_stratum[0]: source_stratum[1]}
    else:
        source_strata = {}
    return _get_transition_flow_connections(
        output_name=output_name,
        source=Compartment.INFECTIOUS,
        dest=Compartment.ON_TREATMENT,
        comps=comps,
        source_strata=source_strata,
    )


def get_incidence_early_connections(comps: List[Compartment], source_stratum=None):
    """
    Track "notifications": flow from infectious to treatment compartment.
    """
    output_name = "incidence_early"
    if source_stratum:
        output_name += "X" + source_stratum[0] + "_" + source_stratum[1]
        source_strata = {source_stratum[0]: source_stratum[1]}
    else:
        source_strata = {}
    return _get_transition_flow_connections(
        output_name=output_name,
        source=Compartment.EARLY_LATENT,
        dest=Compartment.INFECTIOUS,
        comps=comps,
        source_strata=source_strata,
    )


def get_incidence_late_connections(comps: List[Compartment], source_stratum=None):
    """
    Track "notifications": flow from infectious to treatment compartment.
    """
    output_name = "incidence_late"
    if source_stratum:
        output_name += "X" + source_stratum[0] + "_" + source_stratum[1]
        source_strata = {source_stratum[0]: source_stratum[1]}
    else:
        source_strata = {}
    return _get_transition_flow_connections(
        output_name=output_name,
        source=Compartment.LATE_LATENT,
        dest=Compartment.INFECTIOUS,
        comps=comps,
        source_strata=source_strata,
    )


def _get_transition_flow_connections(
    output_name: str,
    source: str,
    dest: str,
    comps: List[Compartment],
    source_strata={},
    dest_strata={},
):
    connections = {}
    connections[output_name] = TransitionFlowOutput(
        source=source,
        dest=dest,
        source_strata=source_strata,
        dest_strata=dest_strata,
    )
    return connections


def get_mortality_flow_infectious(comps: List[Compartment], source_stratum=None):
    output_name = "mortality_infectious"
    connections = {}
    if source_stratum:
        output_name += "X" + source_stratum[0] + "_" + source_stratum[1]
        source_strata = {source_stratum[0]: source_stratum[1]}
    else:
        source_strata = {}
    connections[output_name] = InfectionDeathFlowOutput(
        source=Compartment.INFECTIOUS, source_strata=source_strata
    )
    return connections


def get_mortality_flow_on_treatment(comps: List[Compartment], source_stratum=None):
    output_name = "mortality_on_treatment"
    connections = {}
    if source_stratum:
        output_name += "X" + source_stratum[0] + "_" + source_stratum[1]
        source_strata = {source_stratum[0]: source_stratum[1]}
    else:
        source_strata = {}

    connections[output_name] = InfectionDeathFlowOutput(
        source=Compartment.ON_TREATMENT, source_strata=source_strata
    )
    return connections


def calculate_incidence(time_idx, model, compartment_values, derived_outputs):
    return (
        (derived_outputs["incidence_early"][time_idx] + derived_outputs["incidence_late"][time_idx])
        / sum(compartment_values)
        * 1.0e5
    )


def calculate_tb_mortality(time_idx, model, compartment_values, derived_outputs):
    return (
        (
            derived_outputs["mortality_infectious"][time_idx]
            + derived_outputs["mortality_on_treatment"][time_idx]
        )
        / sum(compartment_values)
        * 1.0e5
    )


def make_stratified_output_func(output_type, source_stratum):
    suffix = "X" + source_stratum[0] + "_" + source_stratum[1]
    outputs_to_sum = {
        "incidence": ["incidence_early", "incidence_late"],
        "mortality": ["mortality_infectious", "mortality_on_treatment"],
    }

    def stratified_output_func(time_idx, model, compartment_values, derived_outputs):
        total_flow = 0
        for output in outputs_to_sum[output_type]:
            total_flow += derived_outputs[output + suffix][time_idx]

        popsize = calculate_stratum_population_size(
            compartment_values, model, [{"name": source_stratum[0], "value": source_stratum[1]}]
        )

        return total_flow / popsize * 1.0e5

    return stratified_output_func


def make_cumulative_output_func(output, start_time_cumul):
    """
    Create a derived output function for cumulative output
    :param output: one of ["cumulative_deaths", "cumulative_diseased"]
    :param start_time_cumul: the reference time
    :return: a derived output function
    """
    base_derived_outputs = {
        "cumulative_diseased": ["incidence_early", "incidence_late"],
        "cumulative_deaths": ["mortality_infectious", "mortality_on_treatment"],
    }

    def calculate_cumulative_output(time_idx, model, compartment_values, derived_outputs):
        if start_time_cumul > max(model.times):
            return 0.0
        ref_time_idx = min(np.where(model.times >= start_time_cumul)[0])
        if time_idx < ref_time_idx:
            return 0.0
        else:
            total = derived_outputs[output][time_idx - 1]
            for relevant_output in base_derived_outputs[output]:
                total += derived_outputs[relevant_output][time_idx]
            return total

    return calculate_cumulative_output


def calculate_population_size(time_idx, model, compartment_values, derived_outputs):
    return sum(compartment_values)


def calculate_case_detection_rate(time_idx, model, compartment_values, derived_outputs):
    screening_rate_func = tanh_based_scaleup(
        model.parameters["time_variant_tb_screening_rate"]["maximum_gradient"],
        model.parameters["time_variant_tb_screening_rate"]["max_change_time"],
        model.parameters["time_variant_tb_screening_rate"]["start_value"],
        model.parameters["time_variant_tb_screening_rate"]["end_value"],
    )
    return screening_rate_func(model.times[time_idx])


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
    relevant_filter_names = [
        s.name
        for s in model.stratifications
        if set(s.compartments) == set(model.original_compartment_names)
    ]

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
            if stratum["name"] in relevant_filter_names:
                if not compartment.has_stratum(stratum["name"], stratum["value"]):
                    relevant = False
        if relevant:
            relevant_compartment_indices.append(i)
    return sum([compartment_values[i] for i in relevant_compartment_indices])


def get_all_derived_output_functions(calculated_outputs, outputs_stratification, model):
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
        "incidence": calculate_incidence,
        "mortality": calculate_tb_mortality,
        "case_detection_rate": calculate_case_detection_rate,
    }
    factory_functions = {
        "prevalence_infectious": make_infectious_prevalence_calculation_function,
        "percentage_latent": make_latency_percentage_calculation_function,
    }
    flow_functions = {
        "notifications": get_notifications_connections,
        "incidence_early": get_incidence_early_connections,
        "incidence_late": get_incidence_late_connections,
        "mortality_infectious": get_mortality_flow_infectious,
        "mortality_on_treatment": get_mortality_flow_on_treatment,
    }
    cumulative_functions = {
        "cumulative_deaths": make_cumulative_output_func,
        "cumulative_diseased": make_cumulative_output_func,
    }
    # need to add two intermediate derived outputs to capture mortality flows
    if "incidence" in calculated_outputs:
        calculated_outputs = ["incidence_early", "incidence_late"] + calculated_outputs
        if "incidence" in outputs_stratification:
            outputs_stratification["incidence_early"] = outputs_stratification["incidence"]
            outputs_stratification["incidence_late"] = outputs_stratification["incidence"]

    # need to add two intermediate derived outputs to capture mortality flows
    if "mortality" in calculated_outputs:
        calculated_outputs = ["mortality_infectious", "mortality_on_treatment"] + calculated_outputs
        if "mortality" in outputs_stratification:
            outputs_stratification["mortality_infectious"] = outputs_stratification["mortality"]
            outputs_stratification["mortality_on_treatment"] = outputs_stratification["mortality"]

    model_stratification_names = [s.name for s in model.stratifications]
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
                        stratification = [
                            model.stratifications[i]
                            for i, s in enumerate(model.stratifications)
                            if s.name == stratification_name
                        ][0]
                        for stratum in stratification.strata:
                            derived_output_functions[
                                requested_output + "X" + stratification_name + "_" + stratum
                            ] = factory_functions[requested_output](
                                [{"name": stratification_name, "value": stratum}]
                            )
        elif requested_output in flow_functions:
            connection = flow_functions[requested_output](model.compartment_names)
            model.add_flow_derived_outputs(connection)
            if requested_output in outputs_stratification:
                for stratification_name in outputs_stratification[requested_output]:
                    if stratification_name in model_stratification_names:
                        stratification = [
                            model.stratifications[i]
                            for i, s in enumerate(model.stratifications)
                            if s.name == stratification_name
                        ][0]
                        for stratum in stratification.strata:
                            connection = flow_functions[requested_output](
                                model.compartment_names, [stratification_name, stratum]
                            )
                            model.add_flow_derived_outputs(connection)
        elif requested_output in cumulative_functions:
            derived_output_functions[requested_output] = cumulative_functions[requested_output](
                requested_output, model.parameters["cumulative_output_start_time"]
            )
        else:
            raise ValueError(
                requested_output
                + " not among simple_functions, factory_functions or flow_functions"
            )

    # stratified incidence and mortality
    for output_type in ["incidence", "mortality"]:
        if output_type in outputs_stratification:
            for stratification_name in outputs_stratification[output_type]:
                if stratification_name in model_stratification_names:
                    stratification = [
                        model.stratifications[i]
                        for i, s in enumerate(model.stratifications)
                        if s.name == stratification_name
                    ][0]
                    for stratum in stratification.strata:  # [stratification_name, stratum]
                        derived_output_functions[
                            f"{output_type}X{stratification_name}_{stratum}"
                        ] = make_stratified_output_func(output_type, [stratification_name, stratum])

    model.add_function_derived_outputs(derived_output_functions)
