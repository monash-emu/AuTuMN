import os
from summer_py.summer_model import StratifiedModel
from summer_py.summer_model.utils.base_compartments import replicate_compartment

from autumn import constants
from autumn.constants import Compartment
from autumn.tb_model import list_all_strata_for_mortality
from autumn.tool_kit.params import load_params
from autumn.tool_kit.scenarios import get_model_times_from_inputs
from autumn.disease_categories.emerging_infections.flows import (
    add_infection_flows,
    add_transition_flows,
    add_recovery_flows,
    add_sequential_compartment_flows,
    add_infection_death_flows,
)
from applications.covid_19.stratification import stratify_by_age, stratify_by_clinical
from applications.covid_19.covid_outputs import (
    find_incidence_outputs,
    create_fully_stratified_incidence_covid,
    calculate_notifications_covid,
)
from applications.covid_19.covid_importation import set_tv_importation_rate
from autumn.demography.social_mixing import load_specific_prem_sheet, update_mixing_with_multipliers
from autumn.demography.population import get_population_size, load_population
from autumn.db import Database

from summer_py.summer_model.strat_model import find_name_components


# Database locations
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DB_PATH = os.path.join(constants.DATA_PATH, "inputs.db")

input_database = Database(database_name=INPUT_DB_PATH)

AUSTRALIA = "australia"
PHILIPPINES = "philippines"
MALAYSIA = "malaysia"
VICTORIA = "victoria"
COUNTRIES = (AUSTRALIA, PHILIPPINES, VICTORIA, MALAYSIA)


def build_covid_model(country: str, update_params: dict):
    """
    Build the master function to run the TB model for Covid-19

    :param update_params: dict
        Any parameters that need to be updated for the current run
    :return: StratifiedModel
        The final model with all parameters and stratifications
    """
    assert country in COUNTRIES, "Invalid country"

    # Get user-requested parameters
    params = load_params(FILE_DIR, application=country)

    model_parameters = params["default"]

    # Update, not used in single application run
    model_parameters.update(update_params)

    # Get population size (by age if age-stratified)
    total_pops, model_parameters = get_population_size(model_parameters, input_database)

    # Replace with Victorian populations
    # total_pops = load_population('31010DO001_201906.XLS', 'Table_6')
    # total_pops = \
    #     [
    #         int(pop) for pop in
    #         total_pops.loc[
    #             (i_pop for i_pop in total_pops.index if 'Persons' in i_pop),
    #             'Victoria'
    #         ]
    #     ]

    all_compartments = [
        Compartment.SUSCEPTIBLE,
        Compartment.RECOVERED,
        Compartment.EXPOSED,
        Compartment.PRESYMPTOMATIC,
        Compartment.INFECTIOUS,
        Compartment.LATE_INFECTIOUS,
    ]

    # Define compartments
    final_compartments, replicated_compartments = [], []
    for compartment in all_compartments:
        if params["default"]["n_compartment_repeats"][compartment] == 1:
            final_compartments.append(compartment)
        else:
            replicated_compartments.append(compartment)

    is_infectious = {
        Compartment.EXPOSED: False,
        Compartment.PRESYMPTOMATIC: True,
        Compartment.INFECTIOUS: True,
        Compartment.LATE_INFECTIOUS: True,
    }

    # Get progression rates from sojourn times, distinguishing to_infectious in order to split this parameter later
    for compartment in params["default"]["compartment_periods"]:
        model_parameters["within_" + compartment] = (
            1.0 / params["default"]["compartment_periods"][compartment]
        )

    # Multiply the progression rates by the number of compartments to keep the average time in exposed the same
    for compartment in is_infectious:
        model_parameters["within_" + compartment] *= float(
            model_parameters["n_compartment_repeats"][compartment]
        )
    model_parameters["to_infectious"] = model_parameters["within_presympt"]

    # Replicate compartments - all repeated compartments are replicated the same number of times, which could be changed
    total_infectious_times = sum(
        [model_parameters["compartment_periods"][comp] for comp in is_infectious]
    )
    infectious_compartments, init_pop = [], {}

    for compartment in is_infectious:
        final_compartments, infectious_compartments, init_pop = replicate_compartment(
            model_parameters["n_compartment_repeats"][compartment],
            final_compartments,
            compartment,
            infectious_compartments,
            init_pop,
            infectious_seed=model_parameters["infectious_seed"]
            * model_parameters["compartment_periods"][compartment]
            / total_infectious_times,
            infectious=is_infectious[compartment],
        )

    # Set integration times
    integration_times = get_model_times_from_inputs(
        round(model_parameters["start_time"]),
        model_parameters["end_time"],
        model_parameters["time_step"],
    )

    # Add flows through replicated compartments
    flows = []
    for compartment in is_infectious:
        flows = add_sequential_compartment_flows(
            flows, model_parameters["n_compartment_repeats"][compartment], compartment
        )

    # Add other flows between compartment types
    flows = add_infection_flows(flows, model_parameters["n_compartment_repeats"]["exposed"])
    flows = add_transition_flows(
        flows,
        model_parameters["n_compartment_repeats"]["exposed"],
        model_parameters["n_compartment_repeats"]["presympt"],
        Compartment.EXPOSED,
        Compartment.PRESYMPTOMATIC,
        "within_exposed",
    )

    # Distinguish to_infectious parameter, so that it can be split later
    model_parameters["to_infectious"] = model_parameters["within_presympt"]
    flows = add_transition_flows(
        flows,
        model_parameters["n_compartment_repeats"]["presympt"],
        model_parameters["n_compartment_repeats"]["infectious"],
        Compartment.PRESYMPTOMATIC,
        Compartment.INFECTIOUS,
        "to_infectious",
    )
    flows = add_transition_flows(
        flows,
        model_parameters["n_compartment_repeats"]["infectious"],
        model_parameters["n_compartment_repeats"]["late"],
        Compartment.INFECTIOUS,
        Compartment.LATE_INFECTIOUS,
        "within_infectious",
    )
    flows = add_recovery_flows(flows, model_parameters["n_compartment_repeats"]["late"])
    flows = add_infection_death_flows(
        flows, model_parameters["n_compartment_repeats"]["infectious"]
    )

    # add importation flows if requested
    if model_parameters["implement_importation"]:
        flows = add_transition_flows(
            flows,
            1,
            model_parameters["n_compartment_repeats"]["exposed"],
            Compartment.SUSCEPTIBLE,
            Compartment.EXPOSED,
            "importation_rate",
        )

    # Get mixing matrix, although would need to adapt this for countries in file _2
    mixing_matrix = load_specific_prem_sheet("all_locations", model_parameters["country"])
    if "mixing_matrix_multipliers" in model_parameters:
        mixing_matrix = update_mixing_with_multipliers(
            mixing_matrix, model_parameters["mixing_matrix_multipliers"]
        )

    # Define output connections to collate
    output_connections = find_incidence_outputs(model_parameters)

    # Define model
    _covid_model = StratifiedModel(
        integration_times,
        final_compartments,
        init_pop,
        model_parameters,
        flows,
        birth_approach="no_birth",
        starting_population=sum(total_pops),
        infectious_compartment=infectious_compartments,
    )

    # set time-variant importation rate
    if model_parameters["implement_importation"]:
        _covid_model = set_tv_importation_rate(
            _covid_model, params["data"]["times_imported_cases"], params["data"]["n_imported_cases"]
        )

    # Stratify model by age without demography
    if "agegroup" in model_parameters["stratify_by"]:
        _covid_model, model_parameters, output_connections = stratify_by_age(
            _covid_model, mixing_matrix, total_pops, model_parameters, output_connections
        )

    # Stratify infectious compartment as high or low infectiousness as requested
    if "clinical" in model_parameters["stratify_by"] and model_parameters["clinical_strata"]:
        _covid_model, model_parameters = stratify_by_clinical(
            _covid_model, model_parameters, final_compartments
        )

    # Add fully stratified incidence to output_connections
    output_connections.update(
        create_fully_stratified_incidence_covid(
            model_parameters["stratify_by"],
            model_parameters["all_stratifications"],
            model_parameters,
        )
    )
    _covid_model.output_connections = output_connections

    # Add notifications to derived_outputs
    _covid_model.derived_output_functions["notifications"] = calculate_notifications_covid
    _covid_model.death_output_categories = list_all_strata_for_mortality(
        _covid_model.compartment_names
    )

    _covid_model.individual_infectiousness_adjustments = [[["late", "clinical_sympt_isolate"], 0.0]]

    print()

    return _covid_model
