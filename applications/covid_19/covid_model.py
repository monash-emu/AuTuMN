import os
import yaml
from summer_py.summer_model import (
    StratifiedModel,
)

from autumn import constants
from autumn.constants import Compartment
from autumn.tb_model import (
    list_all_strata_for_mortality,
)
from autumn.tool_kit.scenarios import get_model_times_from_inputs
from autumn.covid_model.flows import \
    add_infection_flows, add_progression_flows, add_recovery_flows, add_within_exposed_flows, \
    add_within_infectious_flows, replicate_compartment, multiply_flow_value_for_multiple_compartments
from autumn.covid_model.stratification import stratify_by_age
from autumn.social_mixing.social_mixing import load_specific_prem_sheet, load_population, load_age_calibration

# Database locations
file_dir = os.path.dirname(os.path.abspath(__file__))
INPUT_DB_PATH = os.path.join(constants.DATA_PATH, "inputs.db")
PARAMS_PATH = os.path.join(file_dir, "params.yml")


def build_covid_model(update_params={}):
    """
    Build the master function to run the TB model for Covid-19

    :param update_params: dict
        Any parameters that need to be updated for the current run
    :return: StratifiedModel
        The final model with all parameters and stratifications
    """

    # Get user-requested parameters
    with open(PARAMS_PATH, "r") as yaml_file:
        params = yaml.safe_load(yaml_file)
    model_parameters = params["default"]

    # Update, not needed for baseline run
    model_parameters.update(update_params)

    # Define single compartments that don't need to be replicated
    compartments = [
        Compartment.SUSCEPTIBLE,
        Compartment.RECOVERED,
    ]

    # Replicate compartments that need to be repeated
    compartments, _, _ = \
        replicate_compartment(
            model_parameters['n_exposed_compartments'], compartments, Compartment.EXPOSED
        )
    compartments, infectious_compartments, init_pop = \
        replicate_compartment(
            model_parameters['n_infectious_compartments'], compartments, Compartment.INFECTIOUS, infectious_seed=model_parameters['infectious_seed']
        )

    # Multiply the progression rate by the number of compartments to keep the average time in exposed the same
    model_parameters = \
        multiply_flow_value_for_multiple_compartments(
            model_parameters, Compartment.EXPOSED, 'progression'
        )
    model_parameters = \
        multiply_flow_value_for_multiple_compartments(
            model_parameters, Compartment.INFECTIOUS, 'recovery'
        )

    # Set integration times
    integration_times = \
        get_model_times_from_inputs(
            model_parameters["start_time"],
            model_parameters["end_time"],
            model_parameters["time_step"]
        )

    # Sequentially add groups of flows to flows list
    flows = []
    flows = add_infection_flows(
        flows, model_parameters['n_exposed_compartments']
    )
    flows = add_within_exposed_flows(
        flows, model_parameters['n_exposed_compartments']
    )
    flows = add_within_infectious_flows(
        flows, model_parameters['n_infectious_compartments']
    )
    flows = add_progression_flows(
        flows, model_parameters['n_exposed_compartments'], model_parameters['n_infectious_compartments']
    )
    flows = add_recovery_flows(
        flows, model_parameters['n_infectious_compartments']
    )

    mixing_matrix = \
        load_specific_prem_sheet(
            'all_locations_1',
            'Australia'
        )
    # Load a population
    population = load_population('31010DO001_201906.XLS', 'Table_6')
    
    #load case by agegroups
    cases_by_age = load_age_calibration()

    

    # Define model
    _covid_model = StratifiedModel(
        integration_times,
        compartments,
        init_pop,
        model_parameters,
        flows,
        birth_approach='no_birth',
        starting_population=model_parameters['start_population'],
        output_connections={},
        death_output_categories=list_all_strata_for_mortality(compartments),
        infectious_compartment=infectious_compartments
    )

    # Stratify model by age without demography
    if 'agegroup' in model_parameters['stratify_by']:
        age_breaks = [str(i_break) for i_break in list(range(0, 80, 5))]
        _covid_model = \
            stratify_by_age(
                _covid_model,
                age_breaks,
                mixing_matrix=mixing_matrix
            )

    return _covid_model
