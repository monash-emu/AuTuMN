import os
from summer.model import StratifiedModel

from autumn.tool_kit.utils import normalise_sequence, convert_list_contents_to_int
from autumn import constants
from autumn.constants import Compartment
from autumn.tb_model import list_all_strata_for_mortality
from autumn.tool_kit.scenarios import get_model_times_from_inputs
from autumn.disease_categories.emerging_infections.flows import (
    add_infection_flows,
    add_transition_flows,
    add_recovery_flows,
    add_sequential_compartment_flows,
    add_infection_death_flows,
)
from autumn.demography.social_mixing import (
    load_specific_prem_sheet,
    update_mixing_with_multipliers,
    get_total_contact_rates_by_age,
)
from autumn.demography.population import get_population_size
from autumn.db import Database
from autumn.summer_related.parameter_adjustments import split_multiple_parameters

from .stratification import stratify_by_clinical
from .outputs import (
    find_incidence_outputs,
    create_fully_stratified_incidence_covid,
    create_fully_stratified_progress_covid,
    calculate_notifications_covid,
    calculate_incidence_icu_covid,
)
from .importation import set_tv_importation_rate, set_tv_importation_as_birth_rates, importation_props_by_age
from .matrices import build_covid_matrices, apply_npi_effectiveness, update_mixing_parameters_for_prayers
from .preprocess import preprocess_params

# Database locations
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DB_PATH = os.path.join(constants.DATA_PATH, "inputs.db")

input_database = Database(database_name=INPUT_DB_PATH)


def build_model(country: str, params: dict, update_params={}):
    """
    Build the master function to run the TB model for Covid-19

    :param update_params: dict
        Any parameters that need to be updated for the current run
    :return: StratifiedModel
        The final model with all parameters and stratifications
    """
    model_parameters = preprocess_params(params, update_params)

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

    # Define compartments with repeats as needed
    all_compartments = [
        Compartment.SUSCEPTIBLE,
        Compartment.EXPOSED,
        Compartment.PRESYMPTOMATIC,
        Compartment.EARLY_INFECTIOUS,
        Compartment.LATE_INFECTIOUS,
        Compartment.RECOVERED,
    ]
    final_compartments, replicated_compartments = [], []
    for compartment in all_compartments:
        if params["n_compartment_repeats"][compartment] == 1:
            final_compartments.append(compartment)
        else:
            replicated_compartments.append(compartment)
    is_infectious = {
        Compartment.EXPOSED: False,
        Compartment.PRESYMPTOMATIC: True,
        Compartment.EARLY_INFECTIOUS: True,
        Compartment.LATE_INFECTIOUS: True,
    }

    # Get progression rates from sojourn times, distinguishing to_infectious in order to split this parameter later
    for compartment in params["compartment_periods"]:
        model_parameters["within_" + compartment] = 1.0 / params["compartment_periods"][compartment]

    # Multiply the progression rates by the number of compartments to keep the average time in exposed the same
    for compartment in is_infectious:
        model_parameters["within_" + compartment] *= float(
            model_parameters["n_compartment_repeats"][compartment]
        )
    for state in ["hospital_early", "icu_early"]:
        model_parameters["within_" + state] *= float(
            model_parameters["n_compartment_repeats"][Compartment.EARLY_INFECTIOUS]
        )
    for state in ["hospital_late", "icu_late"]:
        model_parameters["within_" + state] *= float(
            model_parameters["n_compartment_repeats"][Compartment.LATE_INFECTIOUS]
        )

    # Distribute infectious seed across infectious compartments
    total_infectious_times = sum(
        [model_parameters["compartment_periods"][comp] for comp in is_infectious]
    )
    init_pop = \
        {comp: model_parameters["infectious_seed"] *
               model_parameters["compartment_periods"][comp] /
               total_infectious_times for
         comp in is_infectious}
    # force the remainder starting population to go to S compartment. Required as entry_compartment is late_infectious
    init_pop[Compartment.SUSCEPTIBLE] = sum(total_pops) - sum(init_pop.values())

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
    flows = add_infection_flows(
        flows, model_parameters["n_compartment_repeats"][Compartment.EXPOSED]
    )
    flows = add_transition_flows(
        flows,
        model_parameters["n_compartment_repeats"][Compartment.EXPOSED],
        model_parameters["n_compartment_repeats"][Compartment.PRESYMPTOMATIC],
        Compartment.EXPOSED,
        Compartment.PRESYMPTOMATIC,
        "within_exposed",
    )

    # Distinguish to_infectious parameter, so that it can be split later
    model_parameters["to_infectious"] = model_parameters["within_presympt"]
    flows = add_transition_flows(
        flows,
        model_parameters["n_compartment_repeats"][Compartment.PRESYMPTOMATIC],
        model_parameters["n_compartment_repeats"][Compartment.EARLY_INFECTIOUS],
        Compartment.PRESYMPTOMATIC,
        Compartment.EARLY_INFECTIOUS,
        "to_infectious",
    )
    flows = add_transition_flows(
        flows,
        model_parameters["n_compartment_repeats"][Compartment.EARLY_INFECTIOUS],
        model_parameters["n_compartment_repeats"][Compartment.LATE_INFECTIOUS],
        Compartment.EARLY_INFECTIOUS,
        Compartment.LATE_INFECTIOUS,
        "within_" + Compartment.EARLY_INFECTIOUS,
    )
    flows = add_recovery_flows(
        flows, model_parameters["n_compartment_repeats"][Compartment.LATE_INFECTIOUS]
    )
    flows = add_infection_death_flows(
        flows, model_parameters["n_compartment_repeats"][Compartment.LATE_INFECTIOUS]
    )

    _birth_approach = 'no_birth'  # may be changed if case importation implemented
    # Add importation flows if requested
    if model_parameters["implement_importation"] and not model_parameters["imported_cases_explict"]:
        flows = add_transition_flows(
            flows,
            1,
            model_parameters["n_compartment_repeats"][Compartment.EXPOSED],
            Compartment.SUSCEPTIBLE,
            Compartment.EXPOSED,
            "import_secondary_rate",
        )
    elif model_parameters["implement_importation"] and model_parameters["imported_cases_explict"]:
        _birth_approach = 'add_crude_birth_rate'

    # Get mixing matrix
    mixing_matrix = load_specific_prem_sheet("all_locations", model_parameters["country"])
    mixing_matrix_multipliers = model_parameters.get("mixing_matrix_multipliers")
    if mixing_matrix_multipliers is not None:
        mixing_matrix = update_mixing_with_multipliers(mixing_matrix, mixing_matrix_multipliers)

    # Define output connections to collate
    output_connections = find_incidence_outputs(model_parameters)

    # Define model
    _covid_model = StratifiedModel(
        integration_times,
        final_compartments,
        init_pop,
        model_parameters,
        flows,
        birth_approach=_birth_approach,
        entry_compartment=Compartment.LATE_INFECTIOUS,  # to model imported cases
        starting_population=sum(total_pops),
        infectious_compartment=[i_comp for i_comp in is_infectious if is_infectious[i_comp]],
    )

    # set time-variant importation rate
    if model_parameters["implement_importation"] and not model_parameters["imported_cases_explict"]:
        _covid_model = set_tv_importation_rate(
            _covid_model, params["data"]["times_imported_cases"], params["data"]["n_imported_cases"]
        )
    elif model_parameters["implement_importation"] and model_parameters["imported_cases_explict"]:
        _covid_model = set_tv_importation_as_birth_rates(
            _covid_model, params["data"]["times_imported_cases"], params["data"]["n_imported_cases"]
        )

    # Stratify model by age
    if "agegroup" in model_parameters["stratify_by"]:
        age_strata = model_parameters["all_stratifications"]["agegroup"]
        adjust_requests = split_multiple_parameters(
            ("to_infectious", "infect_death", "within_late"), age_strata
        )  # Split unchanged parameters for later adjustment

        if model_parameters["implement_importation"] and not model_parameters["imported_cases_explict"]:
            adjust_requests.update(
                {
                    "import_secondary_rate": get_total_contact_rates_by_age(
                        mixing_matrix, direction="horizontal"
                    )
                }
            )

        # Adjust susceptibility for children
        adjust_requests.update(
            {
                "contact_rate": {
                    key: value
                    for key, value in zip(
                        model_parameters["reduced_susceptibility_agegroups"],
                        [model_parameters["young_reduced_susceptibility"]]
                        * len(model_parameters["reduced_susceptibility_agegroups"]),
                    )
                }
            }
        )

        _covid_model.stratify(
            "agegroup",  # Don't use the string age, to avoid triggering automatic demography
            convert_list_contents_to_int(age_strata),
            [],  # Apply to all compartments
            {
                i_break: prop for i_break, prop in zip(age_strata, normalise_sequence(total_pops))
            },  # Distribute starting population
            mixing_matrix=mixing_matrix,
            adjustment_requests=adjust_requests,
            verbose=False,
            entry_proportions=importation_props_by_age
        )

    # Stratify infectious compartment by clinical status
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
    output_connections.update(
        create_fully_stratified_progress_covid(
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
    _covid_model.derived_output_functions["incidence_icu"] = calculate_incidence_icu_covid

    # Do mixing matrix stuff
    mixing_instructions = model_parameters.get("mixing")
    if mixing_instructions:
        if "npi_effectiveness" in model_parameters:
            mixing_instructions = apply_npi_effectiveness(
                mixing_instructions, model_parameters.get("npi_effectiveness")
            )
        if model_parameters['reinstall_regular_prayers']:
            mixing_instructions = update_mixing_parameters_for_prayers(
                mixing_instructions, model_parameters['prayers_params']['restart_time'],
                model_parameters['prayers_params']['prop_participating'],
                model_parameters['prayers_params']['contact_multiplier'],
                t_end=model_parameters['end_time']
            )
        _covid_model.find_dynamic_mixing_matrix = build_covid_matrices(
            model_parameters["country"], mixing_instructions
        )
        _covid_model.dynamic_mixing_matrix = True

    return _covid_model
