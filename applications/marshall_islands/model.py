import os

from summer_py.summer_model import StratifiedModel

from autumn import constants
from autumn.constants import Compartment
from autumn.tb_model.outputs import (
    create_request_stratified_incidence,
    create_request_stratified_notifications,
)
from autumn.tb_model.parameters import add_time_variant_parameter_to_model, build_scale_up_function
from autumn.db import Database
from autumn.tb_model.flows import (
    add_case_detection,
    add_latency_progression,
    add_acf,
    add_acf_ltbi,
    add_treatment_flows,
)
from autumn.tb_model.latency_params import (
    update_transmission_parameters,
    manually_create_age_specific_latency_parameters,
)
from autumn.tb_model.case_detection_params import find_organ_specific_cdr
from autumn.tb_model.stratification import (
    stratify_by_age,
    stratify_by_diabetes,
    stratify_by_organ,
    stratify_by_location,
)
from autumn.tb_model import (
    add_standard_latency_flows,
    add_standard_natural_history_flows,
    add_standard_infection_flows,
    add_birth_rate_functions,
    list_all_strata_for_mortality,
)
from autumn.tool_kit.scenarios import get_model_times_from_inputs

# Database locations
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DB_PATH = os.path.join(constants.DATA_PATH, "inputs.db")


def build_model(params: dict, update_params={}):
    """
    Build the master function to run the TB model for the Republic of the Marshall Islands

    :param update_params: dict
        Any parameters that need to be updated for the current run
    :return: StratifiedModel
        The final model with all parameters and stratifications
    """
    input_database = Database(database_name=INPUT_DB_PATH)

    # Define compartments and initial conditions.
    compartments = [
        Compartment.SUSCEPTIBLE,
        Compartment.EARLY_LATENT,
        Compartment.LATE_LATENT,
        Compartment.INFECTIOUS,
        Compartment.ON_TREATMENT,
        Compartment.RECOVERED,
        # Compartment.LTBI_TREATED,
    ]
    init_pop = {Compartment.INFECTIOUS: 10, Compartment.LATE_LATENT: 100}

    model_parameters = params
    model_parameters.update(update_params)

    # Update partial immunity/susceptibility parameters
    model_parameters = update_transmission_parameters(
        model_parameters, [Compartment.RECOVERED, Compartment.LATE_LATENT, Compartment.LTBI_TREATED]
    )

    # Set integration times
    integration_times = get_model_times_from_inputs(
        model_parameters["start_time"], model_parameters["end_time"], model_parameters["time_step"]
    )

    # Sequentially add groups of flows to flows list
    flows = add_standard_infection_flows([])
    flows = add_standard_latency_flows(flows)
    flows = add_standard_natural_history_flows(flows)
    # flows = add_latency_progression(flows)
    flows = add_case_detection(flows, compartments)
    flows = add_treatment_flows(flows)
    # flows = add_acf(flows, compartments)
    # flows = add_acf_ltbi(flows)

    # Make sure incidence and notifications are tracked during integration
    out_connections = {}
    out_connections.update(
        create_request_stratified_incidence(
            model_parameters["incidence_stratification"], model_parameters["all_stratifications"]
        )
    )
    out_connections.update(
        create_request_stratified_notifications(
            model_parameters["notification_stratifications"],
            model_parameters["all_stratifications"],
        )
    )

    # Define model
    tb_model = StratifiedModel(
        integration_times,
        compartments,
        init_pop,
        model_parameters,
        flows,
        birth_approach="add_crude_birth_rate",
        starting_population=model_parameters["start_population"],
        output_connections=out_connections,
        death_output_categories=list_all_strata_for_mortality(compartments),
    )

    # Add crude birth rate from UN estimates (using Federated States of Micronesia as a proxy as no data for RMI)
    tb_model = add_birth_rate_functions(tb_model, input_database, "FSM")

    # Find raw case detection rate with multiplier, which is 1 by default, and adjust for differences by organ status
    cdr_scaleup_raw = build_scale_up_function(
        model_parameters["cdr"], model_parameters["cdr_multiplier"]
    )
    detect_rate_by_organ = find_organ_specific_cdr(
        cdr_scaleup_raw,
        model_parameters,
        model_parameters["all_stratifications"]["organ"],
        target_organ_props=model_parameters["target_organ_props"],
    )

    # Find base case detection rate and time-variant treatment completion function
    base_detection_rate = detect_rate_by_organ[
        "smearpos" if "organ" in model_parameters["stratify_by"] else "overall"
    ]
    treatment_success_rate = (
        lambda time: build_scale_up_function(model_parameters["tsr"])(time)
        / model_parameters["treatment_duration"]
    )
    treatment_nonsuccess_rate = (
        lambda time: (1.0 - build_scale_up_function(model_parameters["tsr"])(time))
        / model_parameters["treatment_duration"]
    )

    # Set acf screening rate using proportion of population reached and duration of intervention
    # acf_screening_rate = -numpy.log(1 - 0.9) / 0.5
    # acf_rate_over_time = progressive_step_function_maker(
    #     2018.2, 2018.7, acf_screening_rate, scaling_time_fraction=0.3
    # )

    # Initialise acf_rate function
    # acf_rate_function = (
    #     lambda t: model_parameters["acf_coverage"]
    #               * (acf_rate_over_time(t))
    #               * model_parameters["acf_sensitivity"]
    # )
    # acf_ltbi_rate_function = (
    #     lambda t: model_parameters["acf_coverage"]
    #               * (acf_rate_over_time(t))
    #               * model_parameters["acf_ltbi_sensitivity"]
    #               * model_parameters["acf_ltbi_efficacy"]
    # )

    # Assign newly created functions to model parameters
    add_time_variant_parameter_to_model(
        tb_model, "case_detection", base_detection_rate, len(model_parameters["stratify_by"])
    )
    add_time_variant_parameter_to_model(
        tb_model, "treatment_success", treatment_success_rate, len(model_parameters["stratify_by"])
    )
    add_time_variant_parameter_to_model(
        tb_model,
        "treatment_nonsuccess",
        treatment_nonsuccess_rate,
        len(model_parameters["stratify_by"]),
    )
    # add_time_variant_parameter_to_model(
    #     tb_model, 'acf_rate', acf_rate_function, len(model_parameters['stratify_by']))
    # add_time_variant_parameter_to_model(
    #     tb_model, 'acf_ltbi_rate', acf_ltbi_rate_function, len(model_parameters['stratify_by']))

    # Stratification processes
    if "age" in model_parameters["stratify_by"]:
        age_specific_latency_parameters = manually_create_age_specific_latency_parameters(
            model_parameters
        )
        tb_model = stratify_by_age(
            tb_model,
            age_specific_latency_parameters,
            input_database,
            model_parameters["all_stratifications"]["age"],
        )
    if "diabetes" in model_parameters["stratify_by"]:
        tb_model = stratify_by_diabetes(
            tb_model,
            model_parameters,
            model_parameters["all_stratifications"]["diabetes"],
            model_parameters["diabetes_target_props"],
            age_specific_prevalence=False,
        )
    if "organ" in model_parameters["stratify_by"]:
        tb_model = stratify_by_organ(
            tb_model,
            model_parameters,
            detect_rate_by_organ,
            model_parameters["all_stratifications"]["organ"],
        )
    if "location" in model_parameters["stratify_by"]:
        tb_model = stratify_by_location(
            tb_model, model_parameters, model_parameters["all_stratifications"]["location"]
        )

    # Capture reported prevalence in Majuro assuming over-reporting (needed for calibration)
    def calculate_reported_majuro_prevalence(model, time):
        true_prev = 0.0
        pop_majuro = 0.0
        for i, compartment in enumerate(model.compartment_names):
            if "majuro" in compartment:
                pop_majuro += model.compartment_values[i]
                if "infectious" in compartment:
                    true_prev += model.compartment_values[i]
        return (
            1.0e5
            * true_prev
            / pop_majuro
            * (1.0 + model_parameters["over_reporting_prevalence_proportion"])
        )

    tb_model.derived_output_functions.update(
        {"reported_majuro_prevalence": calculate_reported_majuro_prevalence}
    )

    return tb_model
