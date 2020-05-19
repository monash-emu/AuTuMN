import os
from summer.model import StratifiedModel

from autumn.tool_kit.utils import normalise_sequence, convert_list_contents_to_int
from autumn import constants
from autumn.constants import Compartment, BirthApproach
from autumn.tb_model import list_all_strata_for_mortality
from autumn.tool_kit.scenarios import get_model_times_from_inputs

from autumn.demography.social_mixing import get_total_contact_rates_by_age
from autumn.db import Database, find_population_by_agegroup
from autumn.summer_related.parameter_adjustments import split_multiple_parameters

from .stratification import stratify_by_clinical
from .outputs import (
    find_incidence_outputs,
    create_fully_stratified_incidence_covid,
    create_fully_stratified_progress_covid,
    calculate_notifications_covid,
    calculate_incidence_icu_covid,
)
from . import preprocess

# Database locations
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DB_PATH = os.path.join(constants.DATA_PATH, "inputs.db")

input_database = Database(database_name=INPUT_DB_PATH)


def build_model(params: dict):
    """
    Build the master function to run the TB model for Covid-19

    :param update_params: dict
        Any parameters that need to be updated for the current run
    :return: StratifiedModel
        The final model with all parameters and stratifications
    """

    # Update parameters stored in dictionaries that need to be modified during calibration
    params = update_dict_params_for_calibration(params)

    # Adjust infection for relative all-cause mortality compared to China,
    # using a single constant: infection-rate multiplier.
    # FIXME: how consistently is this used?
    ifr_multiplier = params.get("ifr_multiplier")
    hospital_inflate = params["hospital_inflate"]
    hospital_props = params["hospital_props"]
    infection_fatality_props = params["infection_fatality_props"]
    if ifr_multiplier:
        infection_fatality_props = [p * ifr_multiplier for p in infection_fatality_props]
        # FIXME: we should never write back to params
        params["infection_fatality_props"] = infection_fatality_props
    if ifr_multiplier and hospital_inflate:
        hospital_props = [p * ifr_multiplier for p in hospital_props]
        # FIXME: we should never write back to params
        params["hospital_props"] = hospital_props

    # Get the agegroup strata breakpoints.
    agegroup_max = params["agegroup_breaks"][0]
    agegroup_step = params["agegroup_breaks"][1]
    agegroup_strata = list(range(0, agegroup_max, agegroup_step))

    # Calculate the country population size by age-group, using UN data
    country_iso3 = params["iso3"]
    total_pops, _ = find_population_by_agegroup(input_database, agegroup_strata, country_iso3)
    total_pops = [int(1000.0 * total_pops[agebreak][-1]) for agebreak in list(total_pops.keys())]
    starting_pop = sum(total_pops)

    # Define compartments
    compartments = [
        Compartment.SUSCEPTIBLE,
        Compartment.EXPOSED,
        Compartment.PRESYMPTOMATIC,
        Compartment.EARLY_INFECTIOUS,
        Compartment.LATE_INFECTIOUS,
        Compartment.RECOVERED,
    ]
    is_infectious = {
        Compartment.EXPOSED: False,
        Compartment.PRESYMPTOMATIC: True,
        Compartment.EARLY_INFECTIOUS: True,
        Compartment.LATE_INFECTIOUS: True,
    }
    # Calculate compartment periods
    # FIXME: Needs tests.
    base_compartment_periods = params["compartment_periods"]
    compartment_periods_calc = params["compartment_periods_calculated"]
    compartment_periods = preprocess.compartments.calc_compartment_periods(
        base_compartment_periods, compartment_periods_calc
    )

    # Get progression rates from sojourn times, distinguishing to_infectious in order to split this parameter later
    time_within_compartment_params = {}
    for compartment in compartment_periods:
        param_key = f"within_{compartment}"
        time_within_compartment_params[param_key] = 1.0 / compartment_periods[compartment]

    # Distribute infectious seed across infectious compartments
    infectious_seed = params["infectious_seed"]
    total_infectious_times = sum([compartment_periods[c] for c in is_infectious])

    init_pop = {
        c: infectious_seed * compartment_periods[c] / total_infectious_times for c in is_infectious
    }
    # force the remainder starting population to go to S compartment. Required as entry_compartment is late_infectious
    init_pop[Compartment.SUSCEPTIBLE] = sum(total_pops) - sum(init_pop.values())

    # Set integration times
    start_time = params["start_time"]
    end_time = params["end_time"]
    time_step = params["time_step"]
    integration_times = get_model_times_from_inputs(round(start_time), end_time, time_step,)

    is_importation_active = params["implement_importation"]
    is_importation_explict = params["imported_cases_explict"]

    # Add compartmental flows
    add_import_flow = is_importation_active and not is_importation_explict
    flows = preprocess.flows.get_flows(add_import_flow=add_import_flow)

    # Choose a birth apprach
    birth_approach = BirthApproach.NO_BIRTH
    if is_importation_active and is_importation_explict:
        birth_approach = BirthApproach.ADD_CRUDE

    # Build mixing matrix.
    # FIXME: unit tests for build_static
    # FIXME: unit tests for build_dynamic
    country = params["country"]
    static_mixing_matrix = preprocess.mixing_matrix.build_static(country, None)
    dynamic_mixing_matrix = None
    dynamic_mixing_params = params["mixing"]
    if dynamic_mixing_params:
        npi_effectiveness_params = params["npi_effectiveness"]
        is_reinstall_regular_prayers = params.get("reinstall_regular_prayers")
        prayers_params = params.get("prayers_params")
        dynamic_mixing_matrix = preprocess.mixing_matrix.build_dynamic(
            country,
            dynamic_mixing_params,
            npi_effectiveness_params,
            is_reinstall_regular_prayers,
            prayers_params,
            end_time,
        )

    # FIXME: Remove params from model_parameters
    model_parameters = {**params, **time_within_compartment_params}
    model_parameters["to_infectious"] = model_parameters["within_presympt"]
    if is_importation_active and not is_importation_explict:
        # FIXME: summer should handle this internally.
        model_parameters["import_secondary_rate"] = "import_secondary_rate"

    # Define model
    model = StratifiedModel(
        integration_times,
        compartments,
        init_pop,
        model_parameters,
        flows,
        birth_approach=birth_approach,
        entry_compartment=Compartment.LATE_INFECTIOUS,  # to model imported cases
        starting_population=sum(total_pops),
        infectious_compartment=[i_comp for i_comp in is_infectious if is_infectious[i_comp]],
    )
    if dynamic_mixing_matrix:
        model.find_dynamic_mixing_matrix = dynamic_mixing_matrix
        model.dynamic_mixing_matrix = True

    # Set time-variant importation rate
    if is_importation_active and is_importation_explict:
        import_times = params["data"]["times_imported_cases"]
        import_cases = params["data"]["n_imported_cases"]
        symptomatic_props_imported = params["symptomatic_props_imported"]
        prop_detected_among_symptomatic_imported = params[
            "prop_detected_among_symptomatic_imported"
        ]
        model.parameters["crude_birth_rate"] = "crude_birth_rate"
        model.time_variants[
            "crude_birth_rate"
        ] = preprocess.importation.get_importation_rate_func_as_birth_rates(
            import_times,
            import_cases,
            symptomatic_props_imported,
            prop_detected_among_symptomatic_imported,
            starting_pop,
        )
    elif is_importation_active:
        param_name = "import_secondary_rate"
        contact_rate = params["contact_rate"]
        self_isolation_effect = params["self_isolation_effect"]
        enforced_isolation_effect = params["enforced_isolation_effect"]
        import_times = params["data"]["times_imported_cases"]
        import_cases = params["data"]["n_imported_cases"]
        model.adaptation_functions[
            "import_secondary_rate"
        ] = preprocess.importation.get_importation_rate_func(
            country,
            import_times,
            import_cases,
            self_isolation_effect,
            enforced_isolation_effect,
            contact_rate,
            starting_pop,
        )

    # Stratify model by age
    # Coerce age breakpoint numbers into strings - all strata are represented as strings.
    agegroup_strata = [str(s) for s in agegroup_strata]
    if "agegroup" in model_parameters["stratify_by"]:
        age_strata = agegroup_strata
        adjust_requests = split_multiple_parameters(
            ("to_infectious", "infect_death", "within_late"), age_strata
        )  # Split unchanged parameters for later adjustment

        if (
            model_parameters["implement_importation"]
            and not model_parameters["imported_cases_explict"]
        ):
            adjust_requests.update(
                {
                    "import_secondary_rate": get_total_contact_rates_by_age(
                        static_mixing_matrix, direction="horizontal"
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

        model.stratify(
            "agegroup",  # Don't use the string age, to avoid triggering automatic demography
            convert_list_contents_to_int(age_strata),
            [],  # Apply to all compartments
            {
                i_break: prop for i_break, prop in zip(age_strata, normalise_sequence(total_pops))
            },  # Distribute starting population
            mixing_matrix=static_mixing_matrix,
            adjustment_requests=adjust_requests,
            verbose=False,
            entry_proportions=preprocess.importation.IMPORTATION_PROPS_BY_AGE,
        )

    # Stratify infectious compartment by clinical status
    if "clinical" in model_parameters["stratify_by"] and model_parameters["clinical_strata"]:
        model_parameters["all_stratifications"] = {"agegroup": agegroup_strata}
        model, model_parameters = stratify_by_clinical(model, model_parameters, compartments)

    # Define output connections to collate
    output_connections = find_incidence_outputs(model_parameters)

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
    model.output_connections = output_connections

    # Add notifications to derived_outputs
    model.derived_output_functions["notifications"] = calculate_notifications_covid
    model.death_output_categories = list_all_strata_for_mortality(model.compartment_names)
    model.derived_output_functions["incidence_icu"] = calculate_incidence_icu_covid

    return model


# MATT REFACTOR
# TODO: Move or delete


def update_dict_params_for_calibration(params):
    """
    Update some specific parameters that are stored in a dictionary but are updated during calibration.
    For example, we may want to update params['default']['compartment_periods']['incubation'] using the parameter
    ['default']['compartment_periods_incubation']
    :param params: dict
        contains the model parameters
    :return: the updated dictionary
    """

    if "n_imported_cases_final" in params:
        params["data"]["n_imported_cases"][-1] = params["n_imported_cases_final"]

    for location in ["school", "work", "home", "other_locations"]:
        if "npi_effectiveness_" + location in params:
            params["npi_effectiveness"][location] = params["npi_effectiveness_" + location]

    for comp_type in [
        "incubation",
        "infectious",
        "late",
        "hospital_early",
        "hospital_late",
        "icu_early",
        "icu_late",
    ]:
        if "compartment_periods_" + comp_type in params:
            params["compartment_periods"][comp_type] = params["compartment_periods_" + comp_type]

    return params
