import os
import copy
import yaml
from summer_py.summer_model import (
    StratifiedModel,
)
from summer_py.summer_model.utils.flowchart import create_flowchart

from autumn import constants
from autumn.constants import Compartment
from autumn.tb_model import (
    list_all_strata_for_mortality,
)
from autumn.tool_kit.scenarios import get_model_times_from_inputs
from applications.covid_19.flows import \
    add_infection_flows, add_progression_flows, add_recovery_flows, add_within_exposed_flows, \
    add_within_infectious_flows, replicate_compartment, multiply_flow_value_for_multiple_compartments
from applications.covid_19.stratification import stratify_by_age
from applications.covid_19.covid_outputs import find_incidence_outputs
from autumn.demography.social_mixing import load_specific_prem_sheet
from autumn.demography.ageing import add_agegroup_breaks

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

    # Australian population sizes
    total_pops = \
        [1464776, 1502644, 1397182, 1421612, 1566792, 1664609, 1703852, 1561686, 1583254, 1581460, 1523557,
         1454332, 1299406, 1188989, 887721, 652671 + 460555 + 486847]

    # Define single compartments that don't need to be replicated
    compartments = [
        Compartment.SUSCEPTIBLE,
        Compartment.RECOVERED,
    ]

    # Get progression rates from sojourn times
    model_parameters['progression'] = 1. / model_parameters['latent_period']
    model_parameters['recovery'] = 1. / model_parameters['infectious_period']

    # Replicate compartments that need to be repeated
    compartments, _, _ = \
        replicate_compartment(
            model_parameters['n_exposed_compartments'],
            compartments,
            Compartment.EXPOSED
        )
    compartments, infectious_compartments, init_pop = \
        replicate_compartment(
            model_parameters['n_infectious_compartments'],
            compartments,
            Compartment.INFECTIOUS,
            infectious_seed=model_parameters['infectious_seed']
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
    model_parameters['to_infectious'] = model_parameters['within_exposed']

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
        flows,
        model_parameters['n_exposed_compartments']
    )
    flows = add_within_exposed_flows(
        flows,
        model_parameters['n_exposed_compartments']
    )
    flows = add_within_infectious_flows(
        flows,
        model_parameters['n_infectious_compartments'],
        Compartment.INFECTIOUS
    )
    flows = add_progression_flows(
        flows,
        model_parameters['n_exposed_compartments'],
        model_parameters['n_infectious_compartments'],
        Compartment.INFECTIOUS,
        'to_infectious'
    )
    flows = add_recovery_flows(
        flows,
        model_parameters['n_infectious_compartments'],
        Compartment.INFECTIOUS
    )

    mixing_matrix = \
        load_specific_prem_sheet(
            'all_locations_1',
            model_parameters['country']
        )

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
        output_connections=output_connections,
        death_output_categories=list_all_strata_for_mortality(compartments),
        infectious_compartment=infectious_compartments
    )

    # Stratify model by age without demography
    if 'agegroup' in model_parameters['stratify_by']:
        params = add_agegroup_breaks(params)
        age_breakpoints = model_parameters['all_stratifications']['agegroup']
        list_of_starting_pops = [i_pop / sum(total_pops) for i_pop in total_pops]
        starting_props = {i_break: prop for i_break, prop in zip(age_breakpoints, list_of_starting_pops)}
        _covid_model = \
            stratify_by_age(
                _covid_model,
                model_parameters['all_stratifications']['agegroup'],
                mixing_matrix,
                starting_props,
            )

    # Stratify infectious compartment as high or low infectiousness as requested
    if 'infectiousness' in model_parameters['stratify_by']:
        progression_props = [0.7] * 16
        to_infectious_adjustments = \
            {'to_infectiousXagegroup_' + i_break:
                 {
                     'low': 1. - prop,
                     'high': prop
                 }
                for i_break, prop in zip(age_breakpoints, progression_props)
             }
        _covid_model.stratify(
            'infectiousness',
            ['high', 'low'],
            infectious_compartments,
            requested_proportions={},
            adjustment_requests=to_infectious_adjustments,
            verbose=False
        )

    return _covid_model
