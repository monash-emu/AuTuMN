import os
import numpy
import yaml

from summer_py.summer_model import (
    StratifiedModel,
)

from autumn import constants
from autumn.constants import Compartment
from autumn.tb_model.outputs import create_request_stratified_incidence
from autumn.curve import scale_up_function
from autumn.db import Database
from autumn.tb_model.flows import \
    add_case_detection, add_latency_progression, add_acf, add_acf_ltbi, add_treatment_flows
from autumn.tb_model.latency_params import \
    update_transmission_parameters, manually_create_age_specific_latency_parameters
from autumn.tb_model.case_detection_params import find_organ_specific_cdr
from autumn.tb_model.stratification import \
    stratify_by_age, stratify_by_diabetes, stratify_by_organ, stratify_by_location
from autumn.tb_model import (
    add_standard_latency_flows,
    add_standard_natural_history_flows,
    add_standard_infection_flows,
    add_birth_rate_functions,
    create_output_connections_for_incidence_by_stratum,
    list_all_strata_for_mortality,
)
from autumn.tool_kit import progressive_step_function_maker
from autumn.tool_kit.scenarios import get_model_times_from_inputs

# Database locations
file_dir = os.path.dirname(os.path.abspath(__file__))
INPUT_DB_PATH = os.path.join(constants.DATA_PATH, "inputs.db")
PARAMS_PATH = os.path.join(file_dir, "params.yml")

# STRATIFY_BY = ["age"]
# STRATIFY_BY = ["age", "location"]
# STRATIFY_BY = ['age', 'organ']
# STRATIFY_BY = ['age', 'diabetes']
# STRATIFY_BY = ['age', 'diabetes', 'organ']
STRATIFY_BY = ["age", "location", 'diabetes', 'organ']

PLOTTED_STRATIFIED_DERIVED_OUTPUTS = (
    ["notifications"]
)  # use ["incidence", "notifications", "mortality"] to get all outputs

ALL_STRATIFICATIONS = {
    "organ": ["smearpos", "smearneg", "extrapul"],
    "age": ["0", "5", "15", "35", "50"],
    "location": ["majuro", "ebeye", "otherislands"],
    "diabetes": ["diabetic", "nodiabetes"],
}


def build_rmi_timevariant_cdr(cdr_multiplier):
    cdr = {1950.0: 0.0, 1980.0: 0.2, 1990.0: 0.3, 2000.0: 0.4, 2010.0: 0.45, 2015: 0.5}
    return scale_up_function(
        cdr.keys(), [c * cdr_multiplier for c in list(cdr.values())], smoothness=0.2, method=5
    )


def build_rmi_timevariant_tsr():
    tsr = {1950.0: 0.0, 1970.0: 0.2, 1994.0: 0.6, 2000.0: 0.85, 2010.0: 0.87, 2016: 0.87}
    return scale_up_function(tsr.keys(), tsr.values(), smoothness=0.2, method=5)


def build_rmi_model(update_params={}):

    input_database = Database(database_name=INPUT_DB_PATH)

    # Define compartments and initial conditions
    compartments = [
        Compartment.SUSCEPTIBLE,
        Compartment.EARLY_LATENT,
        Compartment.LATE_LATENT,
        Compartment.INFECTIOUS,
        Compartment.ON_TREATMENT,
        Compartment.RECOVERED,
        Compartment.LTBI_TREATED,
    ]
    init_pop = {
        Compartment.INFECTIOUS: 10,
        Compartment.LATE_LATENT: 100
    }

    # Get user-requested parameters
    with open(PARAMS_PATH, "r") as yaml_file:
        params = yaml.safe_load(yaml_file)
    model_parameters = params["default"]

    # Update, not needed for baseline run
    model_parameters.update(
        update_params
    )

    # Update partial immunity/susceptibility parameters
    model_parameters = \
        update_transmission_parameters(
            model_parameters,
            [
                Compartment.RECOVERED,
                Compartment.LATE_LATENT,
                Compartment.LTBI_TREATED
            ]
        )

    # Set integration times
    integration_times = \
        get_model_times_from_inputs(
            model_parameters["start_time"],
            model_parameters["end_time"],
            model_parameters["time_step"]
        )

    # Sequentially add groups of flows to flows list
    flows = add_standard_infection_flows([])
    flows = add_standard_latency_flows(flows)
    flows = add_standard_natural_history_flows(flows)
    flows = add_latency_progression(flows)
    flows = add_case_detection(flows, compartments)
    flows = add_treatment_flows(flows, compartments)
    flows = add_acf(flows, compartments)
    flows = add_acf_ltbi(flows)

    # Make sure incidence is tracked during integration
    out_connections = \
        create_request_stratified_incidence(STRATIFY_BY, ALL_STRATIFICATIONS) \
            if "incidence" in PLOTTED_STRATIFIED_DERIVED_OUTPUTS \
            else {}

    # Define model
    _tb_model = StratifiedModel(
        integration_times,
        compartments,
        init_pop,
        model_parameters,
        flows,
        birth_approach="add_crude_birth_rate",
        starting_population=model_parameters["start_population"],
        output_connections=out_connections,
    )

    # Add crude birth rate from UN estimates (using Federated States of Micronesia as a proxy as no data for RMI)
    _tb_model = add_birth_rate_functions(_tb_model, input_database, "FSM")

    # Find raw case detection rate and adjust for differences by organ status
    cdr_scaleup_raw = build_rmi_timevariant_cdr(model_parameters["cdr_multiplier"])
    detect_rate_by_organ = \
        find_organ_specific_cdr(
            cdr_scaleup_raw,
            model_parameters,
            ALL_STRATIFICATIONS['organ'],
            target_organ_props=
            {
                'smearpos': 0.5,
                'smearneg': 0.3,
                'extrapul': 0.2
            }
        )

    # load time-variant treatment success rate
    treatment_rate = model_parameters['treatment_rate']
    def treatment_completion_rate(time):
        return build_rmi_timevariant_tsr()(time) * treatment_rate

    # tb control recovery rate (detection and treatment) function set for overall if not organ-specific,
    # smearpos otherwise
    tb_control_recovery_rate = lambda t: detect_rate_by_organ['smearpos' if 'organ' in STRATIFY_BY else "overall"](t)

    # set acf screening rate using proportion of population reached and duration of intervention
    acf_screening_rate = -numpy.log(1 - 0.9) / 0.5
    acf_rate_over_time = progressive_step_function_maker(
        2018.2, 2018.7, acf_screening_rate, scaling_time_fraction=0.3
    )

    # initialise acf_rate function
    acf_rate_function = (
        lambda t: model_parameters["acf_coverage"]
                  * (acf_rate_over_time(t))
                  * model_parameters["acf_sensitivity"]
    )
    acf_ltbi_rate_function = (
        lambda t: model_parameters["acf_coverage"]
                  * (acf_rate_over_time(t))
                  * model_parameters["acf_ltbi_sensitivity"]
                  * model_parameters["acf_ltbi_efficacy"]
    )

    # assign newly created functions to model parameters
    if len(STRATIFY_BY) == 0:
        _tb_model.time_variants["case_detection"] = tb_control_recovery_rate
        _tb_model.time_variants["acf_rate"] = acf_rate_function
        _tb_model.time_variants["acf_ltbi_rate"] = acf_ltbi_rate_function
        _tb_model.time_variants['treatment_rate'] = treatment_completion_rate
    else:
        _tb_model.adaptation_functions["case_detection"] = tb_control_recovery_rate
        _tb_model.parameters["case_detection"] = "case_detection"
        _tb_model.parameters['treatment_rate'] = 'treatment_rate'
        _tb_model.adaptation_functions['treatment_rate'] = treatment_completion_rate
        _tb_model.adaptation_functions["acf_rate"] = acf_rate_function
        _tb_model.parameters["acf_rate"] = "acf_rate"
        _tb_model.adaptation_functions["acf_ltbi_rate"] = acf_ltbi_rate_function
        _tb_model.parameters["acf_ltbi_rate"] = "acf_ltbi_rate"

    # Stratification processes
    if "age" in STRATIFY_BY:
        age_specific_latency_parameters = \
            manually_create_age_specific_latency_parameters(
                model_parameters
            )
        _tb_model = \
            stratify_by_age(
                _tb_model, age_specific_latency_parameters, input_database, ALL_STRATIFICATIONS['age']
            )
    if "diabetes" in STRATIFY_BY:
        diab_target_props = {
            0: 0.01,
            5: 0.05,
            15: 0.2,
            35: 0.4,
            50: 0.7
        }
        diabetes_target_props = {}
        for age_group in ALL_STRATIFICATIONS['age']:
            diabetes_target_props.update({
                'age_' + age_group: {'diabetic': diab_target_props[int(age_group)]}
            })
        _tb_model = stratify_by_diabetes(
            _tb_model, model_parameters, ALL_STRATIFICATIONS['diabetes'], diabetes_target_props
        )
    if "organ" in STRATIFY_BY:
        _tb_model = stratify_by_organ(
            _tb_model, model_parameters, detect_rate_by_organ, ALL_STRATIFICATIONS['organ']
        )
    if "location" in STRATIFY_BY:
        _tb_model = stratify_by_location(_tb_model, model_parameters, ALL_STRATIFICATIONS['location'])

    def calculate_reported_majuro_prevalence(model, time):
        if "location" not in STRATIFY_BY:
            return 0.0
        actual_prev = 0.0
        pop_majuro = 0.0
        for i, compartment in enumerate(model.compartment_names):
            if "majuro" in compartment:
                pop_majuro += model.compartment_values[i]
                if "infectious" in compartment:
                    actual_prev += model.compartment_values[i]
        return (
                1.0e5
                * actual_prev
                / pop_majuro
                * (1.0 + model_parameters["over_reporting_prevalence_proportion"])
        )

    _tb_model.derived_output_functions.update(
        {"reported_majuro_prevalence": calculate_reported_majuro_prevalence}
    )

    # add some optional disaggregated derived outputs
    if "notifications" in PLOTTED_STRATIFIED_DERIVED_OUTPUTS:
        # build derived outputs for notifications
        def notification_function_builder(stratum):
            """
                example of stratum: "Xage_0Xstrain_mdr"
            """

            def calculate_notifications(model, time):

                total_notifications = 0.0
                dict_flows = model.transition_flows_dict
                compartment_name = "infectious" + stratum
                comp_idx = model.compartment_idx_lookup[compartment_name]

                infectious_pop = model.compartment_values[comp_idx]
                detection_indices = [
                    index
                    for index, val in dict_flows["parameter"].items()
                    if "case_detection" in val
                ]
                flow_index = [
                    index
                    for index in detection_indices
                    if dict_flows["origin"][index] == model.compartment_names[comp_idx]
                ][0]
                param_name = dict_flows["parameter"][flow_index]
                return total_notifications

            return calculate_notifications

        for compartment in _tb_model.compartment_names:
            if "infectious" in compartment:
                stratum = compartment.split("infectious")[1]
                _tb_model.derived_output_functions[
                    "notifications" + stratum
                    ] = notification_function_builder(stratum)

        # create some personalised notification outputs
        def aggregated_notification_function_builder(location):
            """
                example of location: "majuro"
            """

            def calculate_aggregated_notifications(model, time):
                total_notifications = 0.0
                for key, value in model.derived_outputs.items():
                    if (
                            "notifications" in key
                            and location in key
                            and "agg_notifications" not in key
                    ):
                        this_time_index = model.times.index(time)
                        total_notifications += value[this_time_index]

                return total_notifications

            return calculate_aggregated_notifications

        for location in ["majuro", "ebeye", "otherislands"]:
            _tb_model.derived_output_functions[
                "agg_notificationsXlocation_" + location
                ] = aggregated_notification_function_builder(location)

    if "incidence" in PLOTTED_STRATIFIED_DERIVED_OUTPUTS:
        # add output_connections for all stratum-specific incidence outputs
        _tb_model.output_connections.update(
            create_output_connections_for_incidence_by_stratum(_tb_model.compartment_names)
        )

    if "mortality" in PLOTTED_STRATIFIED_DERIVED_OUTPUTS:
        # prepare death outputs for all strata
        _tb_model.death_output_categories = list_all_strata_for_mortality(
            _tb_model.compartment_names
        )

    return _tb_model


