import os
from summer.model import StratifiedModel

from autumn.tool_kit.utils import normalise_sequence
from autumn import constants
from autumn.constants import Compartment, BirthApproach
from autumn.tb_model import list_all_strata_for_mortality
from autumn.tool_kit.scenarios import get_model_times_from_inputs
from autumn.tool_kit import schema_builder as sb

from autumn.demography.social_mixing import get_total_contact_rates_by_age
from autumn.db import Database, find_population_by_agegroup

from .stratification import stratify_by_clinical
from . import outputs
from . import preprocess

# Database locations
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DB_PATH = os.path.join(constants.DATA_PATH, "inputs.db")

input_database = Database(database_name=INPUT_DB_PATH)


validate_params = sb.build_validator(
    stratify_by=sb.List(str),
    # Country info
    country=str,
    iso3=str,
    # Running time.
    start_time=float,
    end_time=float,
    time_step=float,
    # Compartment construction
    compartment_periods=sb.DictGeneric(str, float),
    compartment_periods_calculated=dict,
    # Infectiousness adjustments (not sure where used)
    ifr_multiplier=float,
    hospital_props=sb.List(float),
    hospital_inflate=bool,
    infection_fatality_props=sb.List(float),
    # Age stratified params
    agegroup_breaks=sb.List(float),
    # Clinical status stratified params
    clinical_strata=sb.List(str),
    non_sympt_infect_multiplier=float,
    hospital_non_icu_infect_multiplier=float,
    icu_infect_multiplier=float,
    icu_mortality_prop=float,
    symptomatic_props=sb.List(float),
    icu_prop=float,
    prop_detected_among_symptomatic=float,
    # Youth reduced susceiptibility adjustment.
    young_reduced_susceptibility=float,
    reduced_susceptibility_agegroups=sb.List(str),
    # Time-variant detection (???)
    tv_detection_b=float,
    tv_detection_c=float,
    tv_detection_sigma=float,
    # Mixing matrix
    mixing=sb.DictGeneric(str, list),
    npi_effectiveness=sb.DictGeneric(str, float),
    reinstall_regular_prayers=bool,
    prayers_params=sb.Dict(restart_time=float, prop_participating=float, contact_multiplier=float,),
    # Something to do with travellers?.
    traveller_quarantine=sb.Dict(times=sb.List(float), values=sb.List(float),),
    # Importation of disease from outside of region.
    implement_importation=bool,
    imported_cases_explict=bool,
    import_secondary_rate=float,
    symptomatic_props_imported=float,
    hospital_props_imported=float,
    icu_prop_imported=float,
    prop_detected_among_symptomatic_imported=float,
    enforced_isolation_effect=float,
    self_isolation_effect=float,
    data=sb.Dict(times_imported_cases=sb.List(float), n_imported_cases=sb.List(float),),
    # Other stuff
    contact_rate=float,
    infect_death=float,
    infectious_seed=int,
    universal_death_rate=float,
)


def build_model(params: dict) -> StratifiedModel:
    """
    Build the master function to run the TB model for Covid-19
    """
    validate_params(params)
    # Update parameters stored in dictionaries that need to be modified during calibration
    params = update_dict_params_for_calibration(params)

    # Adjust infection for relative all-cause mortality compared to China,
    # using a single constant: infection-rate multiplier.
    # FIXME: how consistently is this used?
    ifr_multiplier = params["ifr_multiplier"]
    hospital_inflate = params["hospital_inflate"]
    hospital_props = params["hospital_props"]
    infection_fatality_props = params["infection_fatality_props"]
    symptomatic_props_imported = params["symptomatic_props_imported"]
    if ifr_multiplier:
        infection_fatality_props = [p * ifr_multiplier for p in infection_fatality_props]
        # FIXME: we should never write back to params
        params["infection_fatality_props"] = infection_fatality_props
    if ifr_multiplier and hospital_inflate:
        hospital_props = [
            min(h_prop * ifr_multiplier, 1.0 - symptomatic_props_imported)
            for h_prop in hospital_props
        ]
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
        import_rate_func = preprocess.importation.get_importation_rate_func_as_birth_rates(
            import_times,
            import_cases,
            symptomatic_props_imported,
            prop_detected_among_symptomatic_imported,
            starting_pop,
        )
        model.parameters["crude_birth_rate"] = "crude_birth_rate"
        model.time_variants["crude_birth_rate"] = import_rate_func

    elif is_importation_active:
        param_name = "import_secondary_rate"
        contact_rate = params["contact_rate"]
        self_isolation_effect = params["self_isolation_effect"]
        enforced_isolation_effect = params["enforced_isolation_effect"]
        import_times = params["data"]["times_imported_cases"]
        import_cases = params["data"]["n_imported_cases"]
        import_rate_func = preprocess.importation.get_importation_rate_func(
            country,
            import_times,
            import_cases,
            self_isolation_effect,
            enforced_isolation_effect,
            contact_rate,
            starting_pop,
        )
        model.parameters["import_secondary_rate"] = "import_secondary_rate"
        model.adaptation_functions["import_secondary_rate"] = import_rate_func

    # Stratify model by age.
    # Coerce age breakpoint numbers into strings - all strata are represented as strings.
    agegroup_strata = [str(s) for s in agegroup_strata]
    # Create parameter adjustment request for age stratifications
    youth_agegroups = params["reduced_susceptibility_agegroups"]
    youth_reduced_susceptibility = params["young_reduced_susceptibility"]
    adjust_requests = {
        # No change, required for further stratification by clinical status.
        "to_infectious": {s: 1 for s in agegroup_strata},
        "infect_death": {s: 1 for s in agegroup_strata},
        "within_late": {s: 1 for s in agegroup_strata},
        # Adjust susceptibility for children
        "contact_rate": {
            str(agegroup): youth_reduced_susceptibility for agegroup in youth_agegroups
        },
    }
    if is_importation_active:
        adjust_requests["import_secondary_rate"] = get_total_contact_rates_by_age(
            static_mixing_matrix, direction="horizontal"
        )

    # Distribute starting population over agegroups
    requested_props = {
        agegroup: prop for agegroup, prop in zip(agegroup_strata, normalise_sequence(total_pops))
    }

    # We use "agegroup" instead of "age", to avoid triggering automatic demography features.
    model.stratify(
        "agegroup",
        agegroup_strata,
        compartment_types_to_stratify=[],  # Apply to all compartments
        requested_proportions=requested_props,
        mixing_matrix=static_mixing_matrix,
        adjustment_requests=adjust_requests,
        # FIXME: This seems awfully a lot like a parameter that should go in a YAML file.
        entry_proportions=preprocess.importation.IMPORTATION_PROPS_BY_AGE,
    )

    # Stratify infectious compartment by clinical status
    if "clinical" in model_parameters["stratify_by"] and model_parameters["clinical_strata"]:
        model_parameters["all_stratifications"] = {"agegroup": agegroup_strata}
        model, model_parameters = stratify_by_clinical(model, model_parameters, compartments)

    # Define output connections to collate
    # Track compartment output connections.
    stratum_names = list(set(["X".join(x.split("X")[1:]) for x in model.compartment_names]))
    incidence_connections = outputs.get_incidence_connections(stratum_names)
    progress_connections = outputs.get_progress_connections(stratum_names)
    model.output_connections = {
        **incidence_connections,
        **progress_connections,
    }
    # Add notifications to derived_outputs
    implement_importation = model.parameters["implement_importation"]
    imported_cases_explict = model.parameters["imported_cases_explict"]
    prop_detected_among_symptomatic_imported = model.parameters[
        "prop_detected_among_symptomatic_imported"
    ]
    model.derived_output_functions["notifications"] = outputs.get_calc_notifications_covid(
        implement_importation,
        imported_cases_explict,
        symptomatic_props_imported,
        prop_detected_among_symptomatic_imported,
    )
    model.derived_output_functions["incidence_icu"] = outputs.calculate_incidence_icu_covid
    model.death_output_categories = list_all_strata_for_mortality(model.compartment_names)
    return model


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
