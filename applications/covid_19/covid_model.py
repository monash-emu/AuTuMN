import os
import yaml
import numpy as np
from summer_py.summer_model import StratifiedModel
from summer_py.summer_model.utils.base_compartments import replicate_compartment

from autumn import constants
from autumn.constants import Compartment
from autumn.tb_model import list_all_strata_for_mortality
from autumn.tool_kit.scenarios import get_model_times_from_inputs
from autumn.disease_categories.emerging_infections.flows import \
    add_infection_flows, add_transition_flows, add_recovery_flows, add_sequential_compartment_flows, \
    multiply_flow_value_for_multiple_compartments, \
    add_infection_death_flows
from applications.covid_19.stratification import stratify_by_age, stratify_by_clinical
from applications.covid_19.covid_outputs import find_incidence_outputs, create_fully_stratified_incidence_covid, \
    calculate_notifications_covid
from autumn.demography.social_mixing import load_specific_prem_sheet, load_all_prem_types
from autumn.demography.population import get_population_size
from autumn.db import Database


# Database locations
file_dir = os.path.dirname(os.path.abspath(__file__))
INPUT_DB_PATH = os.path.join(constants.DATA_PATH, 'inputs.db')
PARAMS_PATH = os.path.join(file_dir, 'params.yml')

input_database = Database(database_name=INPUT_DB_PATH)


def build_covid_model(update_params={}):
    """
    Build the master function to run the TB model for Covid-19

    :param update_params: dict
        Any parameters that need to be updated for the current run
    :return: StratifiedModel
        The final model with all parameters and stratifications
    """

    # Get user-requested parameters
    with open(PARAMS_PATH, 'r') as yaml_file:
        params = yaml.safe_load(yaml_file)
    model_parameters = params['default']

    # Update, not used in single application run
    model_parameters.update(update_params)

    # Get population size (by age if age-stratified)
    total_pops, model_parameters = get_population_size(model_parameters, input_database)

    total_pops = [int(t * .26) for t in total_pops]

    # Define compartments
    compartments = [
        Compartment.SUSCEPTIBLE,
        Compartment.RECOVERED,
    ]
    infected_compartment_types = [
        Compartment.EXPOSED,
        Compartment.PRESYMPTOMATIC,
        Compartment.INFECTIOUS
    ]
    n_repeats = model_parameters['n_compartment_repeats']

    # Get progression rates from sojourn times, distinguishing to_infectious in order to split this parameter later
    for compartment in infected_compartment_types + ['hospital', 'icu']:
        model_parameters['within_' + compartment] = 1. / model_parameters[compartment + '_period']
    model_parameters['to_infectious'] = 1. / model_parameters['within_presympt']

    # Multiply the progression rates by the number of compartments to keep the average time in exposed the same
    for compartment in infected_compartment_types:
        model_parameters = \
            multiply_flow_value_for_multiple_compartments(
                model_parameters, compartment, 'within_' + compartment
            )

    # Replicate compartments - all repeated compartments are replicated the same number of times, which could be changed
    total_infectious_times = 0.
    for compartment in infected_compartment_types:
        total_infectious_times += model_parameters[compartment + '_period']
    infectious_compartments, init_pop = [], {}
    is_infectious = {
        Compartment.EXPOSED: False,
        Compartment.PRESYMPTOMATIC: True,
        Compartment.INFECTIOUS: True
    }
    for compartment in infected_compartment_types:
        compartments, infectious_compartments, init_pop = \
            replicate_compartment(
                n_repeats,
                compartments,
                compartment,
                infectious_compartments,
                init_pop,
                infectious_seed=model_parameters['infectious_seed']
                                * model_parameters[compartment + '_period'] /
                                total_infectious_times,
                infectious=is_infectious[compartment]
            )

    # Set integration times
    integration_times = \
        get_model_times_from_inputs(
            round(model_parameters['start_time']),
            model_parameters['end_time'],
            model_parameters['time_step']
        )

    # Sequentially add groups of flows to flows list
    flows = add_infection_flows(
        [], n_repeats
    )
    for compartment in infected_compartment_types:
        flows = add_sequential_compartment_flows(
            flows,
            n_repeats,
            compartment
        )
    flows = add_transition_flows(
        flows,
        n_repeats,
        Compartment.EXPOSED,
        Compartment.PRESYMPTOMATIC,
        'within_exposed'
    )

    # Distinguish to_infectious parameter, so that it can be split later
    model_parameters['to_infectious'] = model_parameters['within_presympt']
    flows = add_transition_flows(
        flows,
        n_repeats,
        Compartment.PRESYMPTOMATIC,
        Compartment.INFECTIOUS,
        'to_infectious'
    )
    flows = add_recovery_flows(flows, n_repeats)
    flows = add_infection_death_flows(flows, n_repeats)

    # Get mixing matrix, although would need to adapt this for countries in file _2
    mixing_matrix = \
        load_specific_prem_sheet(
            'all_locations',
            model_parameters['country']
        )

    # Define output connections to collate
    output_connections = find_incidence_outputs(model_parameters)

    # Define model
    _covid_model = StratifiedModel(
        integration_times,
        compartments,
        init_pop,
        model_parameters,
        flows,
        birth_approach='no_birth',
        starting_population=sum(total_pops),
        infectious_compartment=infectious_compartments
    )

    # Stratify model by age without demography
    if 'agegroup' in model_parameters['stratify_by']:
        _covid_model, model_parameters, output_connections = \
            stratify_by_age(
                _covid_model,
                mixing_matrix,
                total_pops,
                model_parameters,
                output_connections
            )

    # Stratify infectious compartment as high or low infectiousness as requested
    if 'clinical' in model_parameters['stratify_by'] and model_parameters['clinical_strata']:
        _covid_model, model_parameters = \
            stratify_by_clinical(
                _covid_model,
                model_parameters,
                compartments
            )

    # Add fully stratified incidence to output_connections
    output_connections.update(
        create_fully_stratified_incidence_covid(
            model_parameters['stratify_by'],
            model_parameters['all_stratifications'],
            n_repeats
        )
    )
    _covid_model.output_connections = output_connections

    # Add notifications to derived_outputs
    _covid_model.derived_output_functions['notifications'] = \
        calculate_notifications_covid
    _covid_model.death_output_categories = \
        list_all_strata_for_mortality(_covid_model.compartment_names)

    return _covid_model
