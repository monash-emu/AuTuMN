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
    add_infection_flows, add_progression_flows, add_recovery_flows, add_within_exposed_flows

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
    model_parameters.update(
        update_params
    )

    # Define compartments and initial conditions
    compartments = [
        Compartment.SUSCEPTIBLE,
        Compartment.INFECTIOUS,
        Compartment.RECOVERED,
    ]
    init_pop = {
        Compartment.INFECTIOUS: 1,
    }

    # Implement n-exposed compartments into SIR model
    if model_parameters['n_exposed_compartments'] == 0:
        compartments += [Compartment.EXPOSED]
    else:
        for i_exposed in range(model_parameters['n_exposed_compartments']):
            compartments += [Compartment.EXPOSED + '_' + str(i_exposed + 1)]

    # Multiply the progression rate by the number of compartments to keep the average time in exposed the same
    model_parameters['within_exposed'] = \
        model_parameters['progression'] \
        * float(model_parameters['n_exposed_compartments'])

    # Set integration times
    integration_times = \
        get_model_times_from_inputs(
            model_parameters["start_time"],
            model_parameters["end_time"],
            model_parameters["time_step"]
        )

    # Sequentially add groups of flows to flows list
    flows = []
    flows = add_infection_flows(flows, model_parameters['n_exposed_compartments'])
    flows = add_within_exposed_flows(flows, model_parameters['n_exposed_compartments'])
    flows = add_progression_flows(flows, model_parameters['n_exposed_compartments'])
    flows = add_recovery_flows(flows)

    # Define model
    _tb_model = StratifiedModel(
        integration_times,
        compartments,
        init_pop,
        model_parameters,
        flows,
        birth_approach='no_birth',
        starting_population=model_parameters["start_population"],
        output_connections={},
        death_output_categories=list_all_strata_for_mortality(compartments)
    )

    _tb_model.transition_flows.to_csv('temp.csv')

    return _tb_model
