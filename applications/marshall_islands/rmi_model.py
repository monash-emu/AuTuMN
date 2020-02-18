import os
from copy import deepcopy

import numpy
import yaml
from summer_py.summer_model import (
    StratifiedModel,
    split_age_parameter,
    create_sloping_step_function,
)
from summer_py.summer_model.utils.parameter_processing import (
    get_parameter_dict_from_function,
    logistic_scaling_function,
)

from autumn import constants
from autumn.curve import scale_up_function
from autumn.db import Database, get_pop_mortality_functions
from autumn.tb_model import (
    scale_relative_risks_for_equivalence,
    provide_aggregated_latency_parameters,
    get_adapted_age_parameters,
    convert_competing_proportion_to_rate,
    add_standard_latency_flows,
    add_standard_natural_history_flows,
    add_density_infection_flows,
    get_birth_rate_functions,
    create_output_connections_for_incidence_by_stratum,
    list_all_strata_for_mortality,
    plot_time_variant_param,
)
from autumn.tool_kit import (
    return_function_of_function,
    progressive_step_function_maker,
    change_parameter_unit,
)

# Database locations
file_dir = os.path.dirname(os.path.abspath(__file__))
INPUT_DB_PATH = os.path.join(constants.DATA_PATH, "inputs.db")
PARAMS_PATH = os.path.join(file_dir, "params.yml")

# STRATIFY_BY = ["age"]
STRATIFY_BY = ["age", "location"]
# STRATIFY_BY = ['age', 'organ']
# STRATIFY_BY = ['age', 'diabetes']
# STRATIFY_BY = ['age', 'diabetes', 'organ']
# STRATIFY_BY = ["age", "diabetes", "organ", "location"]

PLOTTED_STRATIFIED_DERIVED_OUTPUTS = (
    []
)  # use ["incidence", "notifications", "mortality"] to get all outputs


def build_rmi_model(update_params={}):
    with open(PARAMS_PATH, "r") as f:
        params = yaml.safe_load(f)

    # Load default parameter values
    external_params = params["default"]
    # Update external_params with new parameter values found in update_params
    external_params.update(update_params)

    model_parameters = {
        "contact_rate": external_params["contact_rate"],
        "contact_rate_recovered": external_params["contact_rate"]
        * external_params["rr_transmission_recovered"],
        "contact_rate_infected": external_params["contact_rate"]
        * external_params["rr_transmission_infected"],
        "contact_rate_ltbi_treated": external_params["contact_rate"]
        * external_params["rr_transmission_ltbi_treated"],
        "recovery": external_params["self_recovery_rate"],
        "infect_death": external_params["tb_mortality_rate"],
        "universal_death_rate": 1.0 / 70.0,
        "case_detection": 0.0,
        "ipt_rate": 0.0,
        "acf_rate": 0.0,
        "acf_ltbi_rate": 0.0,
        "crude_birth_rate": 35.0 / 1e3,
    }

    input_database = Database(database_name=INPUT_DB_PATH)
    n_iter = (
        int(
            round(
                (external_params["end_time"] - external_params["start_time"])
                / external_params["time_step"]
            )
        )
        + 1
    )

    integration_times = numpy.linspace(
        external_params["start_time"], external_params["end_time"], n_iter
    ).tolist()

    model_parameters.update(change_parameter_unit(provide_aggregated_latency_parameters(), 365.251))

    # sequentially add groups of flows
    flows = add_density_infection_flows([])
    flows = add_standard_latency_flows(flows)
    flows = add_standard_natural_history_flows(flows)

    # compartments
    compartments = [
        "susceptible",
        "early_latent",
        "late_latent",
        "infectious",
        "recovered",
        "ltbi_treated",
    ]

    # derived output definition
    out_connections = {
        "incidence_early": {"origin": "early_latent", "to": "infectious"},
        "incidence_late": {"origin": "late_latent", "to": "infectious"},
    }

    if "incidence" in PLOTTED_STRATIFIED_DERIVED_OUTPUTS:

        all_stratifications = {
            "organ": ["smearpos", "smearneg", "extrapul"],
            "age": ["0", "5", "15", "35", "50"],
            "location": ["majuro", "ebeye", "otherislands"],
            "diabetes": ["diabetic", "nodiabetes"],
        }

        #  create derived outputs for disaggregated incidence
        for stratification in STRATIFY_BY:
            for stratum in all_stratifications[stratification]:
                for stage in ["early", "late"]:
                    out_connections["incidence_" + stage + "X" + stratification + "_" + stratum] = {
                        "origin": stage + "_latent",
                        "to": "infectious",
                        "to_condition": stratification + "_" + stratum,
                    }

    init_pop = {"infectious": 3, "late_latent": 50}

    # define model     #replace_deaths  add_crude_birth_rate
    _tb_model = StratifiedModel(
        integration_times,
        compartments,
        init_pop,
        model_parameters,
        flows,
        birth_approach="add_crude_birth_rate",
        starting_population=external_params["start_population"],
        output_connections=out_connections,
    )

    # add crude birth rate from un estimates (using Federated States of Micronesia as a proxy as no data for RMI)
    _tb_model = get_birth_rate_functions(_tb_model, input_database, "FSM")

    # add case detection process to basic model
    _tb_model.add_transition_flow(
        {
            "type": "standard_flows",
            "parameter": "case_detection",
            "origin": "infectious",
            "to": "recovered",
        }
    )

    # add ltbi treated infection flow
    _tb_model.add_transition_flow(
        {
            "type": "infection_density",
            "parameter": "contact_rate_ltbi_treated",
            "origin": "ltbi_treated",
            "to": "early_latent",
        }
    )

    # add ACF flow
    _tb_model.add_transition_flow(
        {
            "type": "standard_flows",
            "parameter": "acf_rate",
            "origin": "infectious",
            "to": "recovered",
        }
    )

    # add LTBI ACF flows
    _tb_model.add_transition_flow(
        {
            "type": "standard_flows",
            "parameter": "acf_ltbi_rate",
            "origin": "early_latent",
            "to": "ltbi_treated",
        }
    )

    _tb_model.add_transition_flow(
        {
            "type": "standard_flows",
            "parameter": "acf_ltbi_rate",
            "origin": "late_latent",
            "to": "ltbi_treated",
        }
    )

    # load time-variant case detection rate
    cdr_scaleup_overall = build_rmi_timevariant_cdr(external_params["cdr_multiplier"])

    # targeted TB prevalence proportions by organ
    prop_smearpos = 0.5
    prop_smearneg = 0.3
    prop_extrapul = 0.2

    # disease duration by organ
    overall_duration = prop_smearpos * 1.6 + 5.3 * (1 - prop_smearpos)
    disease_duration = {
        "smearpos": 1.6,
        "smearneg": 5.3,
        "extrapul": 5.3,
        "overall": overall_duration,
    }

    # work out the CDR for smear-positive TB
    def cdr_smearpos(time):
        return cdr_scaleup_overall(time) / (
            prop_smearpos
            + prop_smearneg * external_params["diagnostic_sensitivity_smearneg"]
            + prop_extrapul * external_params["diagnostic_sensitivity_extrapul"]
        )

    def cdr_smearneg(time):
        return cdr_smearpos(time) * external_params["diagnostic_sensitivity_smearneg"]

    def cdr_extrapul(time):
        return cdr_smearpos(time) * external_params["diagnostic_sensitivity_extrapul"]

    cdr_by_organ = {
        "smearpos": cdr_smearpos,
        "smearneg": cdr_smearneg,
        "extrapul": cdr_extrapul,
        "overall": cdr_scaleup_overall,
    }
    detect_rate_by_organ = {}
    for organ in ["smearpos", "smearneg", "extrapul", "overall"]:
        prop_to_rate = convert_competing_proportion_to_rate(1.0 / disease_duration[organ])
        detect_rate_by_organ[organ] = return_function_of_function(cdr_by_organ[organ], prop_to_rate)

    # load time-variant treatment success rate
    rmi_tsr = build_rmi_timevariant_tsr()

    # create a treatment success rate function adjusted for treatment support intervention
    tsr_function = lambda t: rmi_tsr(t)

    # tb control recovery rate (detection and treatment) function set for overall if not organ-specific, smearpos otherwise
    if "organ" not in STRATIFY_BY:
        tb_control_recovery_rate = lambda t: tsr_function(t) * detect_rate_by_organ["overall"](t)
    else:
        tb_control_recovery_rate = lambda t: tsr_function(t) * detect_rate_by_organ["smearpos"](t)

    # set acf screening rate using proportion of population reached and duration of intervention
    acf_screening_rate = -numpy.log(1 - 0.90) / 0.5

    acf_rate_over_time = progressive_step_function_maker(
        2018.2, 2018.7, acf_screening_rate, scaling_time_fraction=0.3
    )

    # initialise acf_rate function
    acf_rate_function = (
        lambda t: external_params["acf_coverage"]
        * (acf_rate_over_time(t))
        * external_params["acf_sensitivity"]
        * (rmi_tsr(t))
    )

    acf_ltbi_rate_function = (
        lambda t: external_params["acf_coverage"]
        * (acf_rate_over_time(t))
        * external_params["acf_ltbi_sensitivity"]
        * external_params["acf_ltbi_efficacy"]
    )

    # time_variant contact_rate to simulate living condition improvement
    contact_rate_function = (
        lambda t: (
            external_params["minimum_tv_beta_multiplier"]
            + (1.0 - external_params["minimum_tv_beta_multiplier"])
            * numpy.exp(-external_params["beta_decay_rate"] * (t - 1900.0))
        )
        * external_params["contact_rate"]
    )

    plot_time_variant_param(contact_rate_function, [1900, 2020])

    # create time-variant functions for the different contact rates # did not get it to work with a loop!!!
    beta_func = lambda t: contact_rate_function(t)
    beta_func_infected = (
        lambda t: contact_rate_function(t) * external_params["rr_transmission_infected"]
    )
    beta_func_recovered = (
        lambda t: contact_rate_function(t) * external_params["rr_transmission_recovered"]
    )
    beta_func_ltbi_treated = (
        lambda t: contact_rate_function(t) * external_params["rr_transmission_ltbi_treated"]
    )

    # assign newly created functions to model parameters
    if len(STRATIFY_BY) == 0:
        _tb_model.time_variants["case_detection"] = tb_control_recovery_rate
        _tb_model.time_variants["acf_rate"] = acf_rate_function
        _tb_model.time_variants["acf_ltbi_rate"] = acf_ltbi_rate_function

        ###################################
        _tb_model.time_variants["contact_rate"] = beta_func
        _tb_model.time_variants["contact_rate_infected"] = beta_func_infected
        _tb_model.time_variants["contact_rate_recovered"] = beta_func_recovered
        _tb_model.time_variants["contact_rate_ltbi_treated"] = beta_func_ltbi_treated
    else:
        _tb_model.adaptation_functions["case_detection"] = tb_control_recovery_rate
        _tb_model.parameters["case_detection"] = "case_detection"

        _tb_model.adaptation_functions["acf_rate"] = acf_rate_function
        _tb_model.parameters["acf_rate"] = "acf_rate"

        _tb_model.adaptation_functions["acf_ltbi_rate"] = acf_ltbi_rate_function
        _tb_model.parameters["acf_ltbi_rate"] = "acf_ltbi_rate"

        ###################################################################################
        _tb_model.adaptation_functions["contact_rate"] = beta_func
        _tb_model.parameters["contact_rate"] = "contact_rate"

        _tb_model.adaptation_functions["contact_rate_infected"] = beta_func_infected
        _tb_model.parameters["contact_rate_infected"] = "contact_rate_infected"

        _tb_model.adaptation_functions["contact_rate_recovered"] = beta_func_recovered
        _tb_model.parameters["contact_rate_recovered"] = "contact_rate_recovered"

        _tb_model.adaptation_functions["contact_rate_ltbi_treated"] = beta_func_ltbi_treated
        _tb_model.parameters["contact_rate_ltbi_treated"] = "contact_rate_ltbi_treated"

    if "age" in STRATIFY_BY:
        age_breakpoints = [0, 5, 15, 35, 50, 70]
        age_infectiousness = get_parameter_dict_from_function(
            logistic_scaling_function(10.0), age_breakpoints
        )
        age_params = get_adapted_age_parameters(age_breakpoints)
        age_params.update(split_age_parameter(age_breakpoints, "contact_rate"))

        # adjustment of latency parameters
        # for param in ['early_progression', 'late_progression']:
        #     for age_break in age_breakpoints:
        #         age_params[param][str(age_break) + 'W'] *= external_params['latency_adjustment']

        pop_morts = get_pop_mortality_functions(
            input_database,
            age_breakpoints,
            country_iso_code="FSM",
            emigration_value=0.01,
            emigration_start_time=1990.0,
        )

        age_params["universal_death_rate"] = {}
        for age_break in age_breakpoints:
            _tb_model.time_variants["universal_death_rateXage_" + str(age_break)] = pop_morts[
                age_break
            ]
            _tb_model.parameters[
                "universal_death_rateXage_" + str(age_break)
            ] = "universal_death_rateXage_" + str(age_break)

            age_params["universal_death_rate"][
                str(age_break) + "W"
            ] = "universal_death_rateXage_" + str(age_break)
        _tb_model.parameters["universal_death_rateX"] = 0.0

        # add BCG effect without stratification assuming constant 100% coverage
        bcg_wane = create_sloping_step_function(15.0, 0.3, 30.0, 1.0)
        age_bcg_efficacy_dict = get_parameter_dict_from_function(
            lambda value: bcg_wane(value), age_breakpoints
        )
        age_params.update({"contact_rate": age_bcg_efficacy_dict})

        _tb_model.stratify(
            "age",
            deepcopy(age_breakpoints),
            [],
            {},
            adjustment_requests=age_params,
            infectiousness_adjustments=age_infectiousness,
            verbose=False,
        )

    if "diabetes" in STRATIFY_BY:
        props_diabetes = {"diabetic": 0.3, "nodiabetes": 0.7}
        progression_adjustments = {
            "diabetic": external_params["rr_progression_diabetic"],
            "nodiabetes": 1.0,
        }

        _tb_model.stratify(
            "diabetes",
            ["diabetic", "nodiabetes"],
            [],
            verbose=False,
            requested_proportions=props_diabetes,
            adjustment_requests={
                "early_progression": progression_adjustments,
                "late_progression": progression_adjustments,
            },
            entry_proportions={"diabetic": 0.01, "nodiabetes": 0.99},
            target_props={
                "age_0": {"diabetic": 0.05},
                "age_5": {"diabetic": 0.1},
                "age_15": {"diabetic": 0.2},
                "age_35": {"diabetic": 0.4},
                "age_50": {"diabetic": 0.5},
                "age_70": {"diabetic": 0.8},
            },
        )

    if "organ" in STRATIFY_BY:
        props_smear = {
            "smearpos": external_params["prop_smearpos"],
            "smearneg": 1.0 - (external_params["prop_smearpos"] + 0.2),
            "extrapul": 0.2,
        }
        mortality_adjustments = {"smearpos": 1.0, "smearneg": 0.064, "extrapul": 0.064}
        recovery_adjustments = {"smearpos": 1.0, "smearneg": 0.56, "extrapul": 0.56}

        # workout the detection rate adjustment by organ status
        adjustment_smearneg = (
            detect_rate_by_organ["smearneg"](2015.0) / detect_rate_by_organ["smearpos"](2015.0)
            if detect_rate_by_organ["smearpos"](2015.0) > 0.0
            else 1.0
        )
        adjustment_extrapul = (
            detect_rate_by_organ["extrapul"](2015.0) / detect_rate_by_organ["smearpos"](2015.0)
            if detect_rate_by_organ["smearpos"](2015.0) > 0.0
            else 1.0
        )

        _tb_model.stratify(
            "organ",
            ["smearpos", "smearneg", "extrapul"],
            ["infectious"],
            infectiousness_adjustments={"smearpos": 1.0, "smearneg": 0.25, "extrapul": 0.0},
            verbose=False,
            requested_proportions=props_smear,
            adjustment_requests={
                "recovery": recovery_adjustments,
                "infect_death": mortality_adjustments,
                "case_detection": {
                    "smearpos": 1.0,
                    "smearneg": adjustment_smearneg,
                    "extrapul": adjustment_extrapul,
                },
                "early_progression": props_smear,
                "late_progression": props_smear,
            },
        )

    if "location" in STRATIFY_BY:
        props_location = {"majuro": 0.523, "ebeye": 0.2, "otherislands": 0.277}

        raw_relative_risks_loc = {"majuro": 1.0}
        for stratum in ["ebeye", "otherislands"]:
            raw_relative_risks_loc[stratum] = external_params["rr_transmission_" + stratum]
        scaled_relative_risks_loc = scale_relative_risks_for_equivalence(
            props_location, raw_relative_risks_loc
        )

        # dummy matrix for mixing by location
        location_mixing = numpy.array([0.9, 0.05, 0.05, 0.05, 0.9, 0.05, 0.05, 0.05, 0.9]).reshape(
            (3, 3)
        )
        location_mixing *= 3.0  # adjusted such that heterogeneous mixing yields similar overall burden as homogeneous

        location_adjustments = {}
        for beta_type in ["", "_infected", "_recovered"]:
            location_adjustments["contact_rate" + beta_type] = scaled_relative_risks_loc

        location_adjustments["case_detection"] = {}
        for stratum in ["majuro", "ebeye", "otherislands"]:
            location_adjustments["case_detection"][stratum] = external_params[
                "case_detection_" + stratum + "_multiplier"
            ]

        location_adjustments["acf_rate"] = {}
        for stratum in ["majuro", "ebeye", "otherislands"]:
            location_adjustments["acf_rate"][stratum] = external_params[
                "acf_" + stratum + "_switch"
            ]

        location_adjustments["acf_ltbi_rate"] = {}
        for stratum in ["majuro", "ebeye", "otherislands"]:
            location_adjustments["acf_ltbi_rate"][stratum] = external_params[
                "acf_ltbi_" + stratum + "_switch"
            ]

        _tb_model.stratify(
            "location",
            ["majuro", "ebeye", "otherislands"],
            [],
            # infectiousness_adjustments={"majuro": 0.9, "ebeye": 1.9, "otherislands": 1.1},
            requested_proportions=props_location,
            verbose=False,
            entry_proportions=props_location,
            adjustment_requests=location_adjustments,
            mixing_matrix=location_mixing,
        )

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
            * (1.0 + external_params["over_reporting_prevalence_proportion"])
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
                detection_tx_rate = model.get_parameter_value(param_name, time)
                tsr = rmi_tsr(time)
                if tsr > 0.0:
                    total_notifications += infectious_pop * detection_tx_rate / tsr

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


def build_rmi_timevariant_cdr(cdr_multiplier):
    cdr = {1950.0: 0.0, 1980.0: 0.10, 1990.0: 0.1, 2000.0: 0.2, 2010.0: 0.3, 2015: 0.4}
    return scale_up_function(
        cdr.keys(), [c * cdr_multiplier for c in list(cdr.values())], smoothness=0.2, method=5
    )


def build_rmi_timevariant_tsr():
    tsr = {1950.0: 0.0, 1970.0: 0.2, 1994.0: 0.6, 2000.0: 0.85, 2010.0: 0.87, 2016: 0.87}
    return scale_up_function(tsr.keys(), tsr.values(), smoothness=0.2, method=5)
