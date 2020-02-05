import os
from datetime import datetime
from time import time
from copy import deepcopy

import numpy
import pandas as pd
from summer_py.summer_model import (
    StratifiedModel,
    split_age_parameter,
    create_sloping_step_function,
    find_name_components,
    find_stem,
)
from summer_py.parameter_processing import (
    get_parameter_dict_from_function,
    logistic_scaling_function,
)

from autumn import constants
from autumn.curve import scale_up_function
from autumn.db import Database, get_pop_mortality_functions
from autumn.tb_model import (
    add_combined_incidence,
    create_output_connections_for_incidence_by_stratum,
    list_all_strata_for_mortality,
    load_model_scenario,
    load_calibration_from_db,
    scale_relative_risks_for_equivalence,
    provide_aggregated_latency_parameters,
    get_adapted_age_parameters,
    convert_competing_proportion_to_rate,
    store_run_models,
    add_standard_latency_flows,
    add_standard_natural_history_flows,
    add_standard_infection_flows,
    get_birth_rate_functions,
    create_multi_scenario_outputs,
    create_mcmc_outputs,
    DummyModel,
)
from autumn.tool_kit import run_multi_scenario, return_function_of_function, change_parameter_unit

# Database locations
file_dir = os.path.dirname(os.path.abspath(__file__))
timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
OUTPUT_DB_PATH = os.path.join(file_dir, "databases", f"outputs_{timestamp}.db")
INPUT_DB_PATH = os.path.join(constants.DATA_PATH, "inputs.db")

STRATIFY_BY = ["age", "strain", "location", "organ"]


def build_mongolia_timevariant_cdr(cdr_multiplier):
    cdr = {1950.0: 0.0, 1980.0: 0.10, 1990.0: 0.15, 2000.0: 0.20, 2010.0: 0.30, 2015: 0.33}
    return scale_up_function(
        cdr.keys(), [c * cdr_multiplier for c in list(cdr.values())], smoothness=0.2, method=5
    )


def build_mongolia_timevariant_tsr():
    tsr = {1950.0: 0.0, 1970.0: 0.2, 1994.0: 0.6, 2000.0: 0.85, 2010.0: 0.87, 2016: 0.9}
    return scale_up_function(tsr.keys(), tsr.values(), smoothness=0.2, method=5)


def build_mongolia_model(update_params={}):
    # some default parameter values
    external_params = {  # run configuration
        "start_time": 1900.0,
        "end_time": 2035.0,
        "time_step": 1.0,
        "start_population": 3000000,
        # base model definition:
        "contact_rate": 14.0,
        "rr_transmission_recovered": 1.0,
        "rr_transmission_infected": 0.21,
        "adult_latency_adjustment": 4.0,  # used to increase adult progression rates due to pollution
        "self_recovery_rate": 0.231,  # this is for smear-positive TB
        "tb_mortality_rate": 0.389,  # this is for smear-positive TB
        "prop_smearpos": 0.5,
        "cdr_multiplier": 1.0,
        # MDR-TB:
        "dr_amplification_prop_among_nonsuccess": 0.20,  # based on Cox et al and Bonnet et al
        "prop_mdr_detected_as_mdr": 0.5,
        "mdr_tsr": 0.6,
        "mdr_infectiousness_multiplier": 1.1,
        # diagnostic sensitivity by organ status:
        "diagnostic_sensitivity_smearpos": 1.0,
        "diagnostic_sensitivity_smearneg": 0.7,
        "diagnostic_sensitivity_extrapul": 0.5,
        # adjustments by location
        "rr_transmission_urban_ger": 3.0,  # reference: rural_province
        "rr_transmission_urban_nonger": 0.8,  # reference: rural_province
        "rr_transmission_prison": 50,  # reference: rural_province
        # IPT
        "ipt_age_0_ct_coverage": 0.17,  # Children contact tracing coverage  .17
        "ipt_age_5_ct_coverage": 0.0,  # Children contact tracing coverage
        "ipt_age_15_ct_coverage": 0.0,  # Children contact tracing coverage
        "ipt_age_60_ct_coverage": 0.0,  # Children contact tracing coverage
        "yield_contact_ct_tstpos_per_detected_tb": 2.0,  # expected number of infections traced per index
        "ipt_efficacy": 0.75,  # based on intention-to-treat
        "ds_ipt_switch": 1.0,  # used as a DS-specific multiplier to the coverage defined above
        "mdr_ipt_switch": 0.0,  # used as an MDR-specific multiplier to the coverage defined above
        # Treatment improvement (C-DOTS)
        "reduction_negative_tx_outcome": 0.0,
        # ACF for risk groups
        "acf_coverage": 0.0,
        "acf_sensitivity": 0.8,
        "acf_rural_province_switch": 0.0,
        "acf_urban_nonger_switch": 0.0,
        "acf_urban_ger_switch": 0.0,
        "acf_prison_switch": 0.0,
    }

    # update external_params with MCMC mle estimates
    mle_estimates = {
        "contact_rate": 13.17359,  # 10.44,
        "adult_latency_adjustment": 2.894561,
        "dr_amplification_prop_among_nonsuccess": 0.1662956,
        "self_recovery_rate": 0.2497045,
        "tb_mortality_rate": 0.3729356,
        "rr_transmission_recovered": 0.959254,
        "cdr_multiplier": 1.077391,
    }
    external_params.update(mle_estimates)

    # update external_params with new parameter values found in update_params
    external_params.update(update_params)

    model_parameters = {
        "contact_rate": external_params["contact_rate"],
        "contact_rate_recovered": external_params["contact_rate"]
        * external_params["rr_transmission_recovered"],
        "contact_rate_infected": external_params["contact_rate"]
        * external_params["rr_transmission_infected"],
        "recovery": external_params["self_recovery_rate"],
        "infect_death": external_params["tb_mortality_rate"],
        "universal_death_rate": 1.0 / 50.0,
        "case_detection": 0.0,
        "ipt_rate": 0.0,
        "acf_rate": 0.0,
        "dr_amplification": 0.0,  # high value for testing
        "crude_birth_rate": 20.0 / 1e3,
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
    flows = add_standard_infection_flows([])
    flows = add_standard_latency_flows(flows)
    flows = add_standard_natural_history_flows(flows)

    # compartments
    compartments = ["susceptible", "early_latent", "late_latent", "infectious", "recovered"]

    # define model     #replace_deaths  add_crude_birth_rate
    init_pop = {"infectious": 1000, "late_latent": 1000000}

    _tb_model = StratifiedModel(
        integration_times,
        compartments,
        init_pop,
        model_parameters,
        flows,
        birth_approach="replace_deaths",
        starting_population=external_params["start_population"],
        output_connections={},
        derived_output_functions={},
        death_output_categories=((), ("age_0",)),
    )

    # add crude birth rate from un estimates
    _tb_model = get_birth_rate_functions(_tb_model, input_database, "MNG")

    # add case detection process to basic model
    _tb_model.add_transition_flow(
        {
            "type": "standard_flows",
            "parameter": "case_detection",
            "origin": "infectious",
            "to": "recovered",
        }
    )

    # Add IPT as a customised flow
    def ipt_flow_func(model, n_flow, _time, _compartment_values):
        """
        Work out the number of detected individuals from the relevant active TB compartments (with regard to the origin
        latent compartment of n_flow) multiplied with the proportion of the relevant infected contacts that is from this
        latent compartment.
        """
        dict_flows = model.transition_flows.to_dict()
        origin_comp_name = dict_flows["origin"][n_flow]
        components_latent_comp = find_name_components(origin_comp_name)

        # find compulsory tags to be found in relevant infectious compartments
        tags = []
        for component in components_latent_comp:
            if "location_" in component or "strain_" in component:
                tags.append(component)

        # loop through all relevant infectious compartments
        total_tb_detected = 0.0
        for comp_ind in model.infectious_indices["all_strains"]:
            active_components = find_name_components(model.compartment_names[comp_ind])
            if all(elem in active_components for elem in tags):
                infectious_pop = _compartment_values[comp_ind]
                detection_indices = [
                    index
                    for index, val in dict_flows["parameter"].items()
                    if "case_detection" in val
                ]
                flow_index = [
                    index
                    for index in detection_indices
                    if dict_flows["origin"][index] == model.compartment_names[comp_ind]
                ][0]
                param_name = dict_flows["parameter"][flow_index]
                detection_tx_rate = model.get_parameter_value(param_name, _time)
                tsr = mongolia_tsr(_time) + external_params["reduction_negative_tx_outcome"] * (
                    1.0 - mongolia_tsr(_time)
                )
                if "strain_mdr" in model.compartment_names[comp_ind]:
                    tsr = external_params["mdr_tsr"] * external_params["prop_mdr_detected_as_mdr"]
                if tsr > 0.0:
                    total_tb_detected += infectious_pop * detection_tx_rate / tsr

        # list all latent compartments relevant to the relevant infectious population
        relevant_latent_compartments_indices = [
            i
            for i, comp_name in enumerate(model.compartment_names)
            if find_stem(comp_name) == "early_latent" and all(elem in comp_name for elem in tags)
        ]

        total_relevant_latent_size = sum(
            _compartment_values[i] for i in relevant_latent_compartments_indices
        )
        current_latent_size = _compartment_values[model.compartment_names.index(origin_comp_name)]
        prop_of_relevant_latent = (
            current_latent_size / total_relevant_latent_size
            if total_relevant_latent_size > 0.0
            else 0.0
        )

        return total_tb_detected * prop_of_relevant_latent

    _tb_model.add_transition_flow(
        {
            "type": "customised_flows",
            "parameter": "ipt_rate",
            "origin": "early_latent",
            "to": "recovered",
            "function": ipt_flow_func,
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

    # load time-variant case detection rate
    cdr_scaleup_overall = build_mongolia_timevariant_cdr(external_params["cdr_multiplier"])

    # targeted TB prevalence proportions by organ
    prop_smearpos = 0.25
    prop_smearneg = 0.40
    prop_extrapul = 0.35

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
        # Had to replace external_params['diagnostic_sensitivity_smearneg'] with its hard-coded value .7 to avoid
        # cdr_smearpos to be affected when increasing diagnostic_sensitivity_smearneg in interventions (e.g. Xpert)

        # return (cdr_scaleup_overall(time) /
        #         (prop_smearpos + prop_smearneg * external_params['diagnostic_sensitivity_smearneg'] +
        #          prop_extrapul * external_params['diagnostic_sensitivity_extrapul']))
        return cdr_scaleup_overall(time) / (
            prop_smearpos
            + prop_smearneg * 0.7
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
    mongolia_tsr = build_mongolia_timevariant_tsr()

    # create a treatment succes rate function adjusted for treatment support intervention
    tsr_function = lambda t: mongolia_tsr(t) + external_params["reduction_negative_tx_outcome"] * (
        1.0 - mongolia_tsr(t)
    )

    # tb control recovery rate (detection and treatment) function set for overall if not organ-specific, smearpos otherwise
    if "organ" not in STRATIFY_BY:
        tb_control_recovery_rate = lambda t: tsr_function(t) * detect_rate_by_organ["overall"](t)
    else:
        tb_control_recovery_rate = lambda t: tsr_function(t) * detect_rate_by_organ["smearpos"](t)

    # initialise ipt_rate function assuming coverage of 1.0 before age stratification
    ipt_rate_function = (
        lambda t: 1.0
        * external_params["yield_contact_ct_tstpos_per_detected_tb"]
        * external_params["ipt_efficacy"]
    )

    # initialise acf_rate function
    acf_rate_function = (
        lambda t: external_params["acf_coverage"]
        * external_params["acf_sensitivity"]
        * (
            mongolia_tsr(t)
            + external_params["reduction_negative_tx_outcome"] * (1.0 - mongolia_tsr(t))
        )
    )

    # assign newly created functions to model parameters
    _tb_model.adaptation_functions["case_detection"] = tb_control_recovery_rate
    _tb_model.parameters["case_detection"] = "case_detection"

    _tb_model.adaptation_functions["ipt_rate"] = ipt_rate_function
    _tb_model.parameters["ipt_rate"] = "ipt_rate"

    _tb_model.adaptation_functions["acf_rate"] = acf_rate_function
    _tb_model.parameters["acf_rate"] = "acf_rate"

    if "strain" in STRATIFY_BY:
        mdr_adjustment = (
            external_params["prop_mdr_detected_as_mdr"] * external_params["mdr_tsr"] / 0.9
        )  # /.9 for last DS TSR

        _tb_model.stratify(
            "strain",
            ["ds", "mdr"],
            ["early_latent", "late_latent", "infectious"],
            verbose=False,
            requested_proportions={"mdr": 0.0},
            adjustment_requests={
                "contact_rate": {"ds": 1.0, "mdr": 1.0},
                "case_detection": {"mdr": mdr_adjustment},
                "ipt_rate": {
                    "ds": 1.0,  # external_params['ds_ipt_switch'],
                    "mdr": external_params["mdr_ipt_switch"],
                },
            },
            infectiousness_adjustments={
                "ds": 1.0,
                "mdr": external_params["mdr_infectiousness_multiplier"],
            },
        )

        _tb_model.add_transition_flow(
            {
                "type": "standard_flows",
                "parameter": "dr_amplification",
                "origin": "infectiousXstrain_ds",
                "to": "infectiousXstrain_mdr",
                "implement": len(_tb_model.all_stratifications),
            }
        )

        dr_amplification_rate = (
            lambda t: detect_rate_by_organ["overall"](t)
            * (1.0 - mongolia_tsr(t))
            * (1.0 - external_params["reduction_negative_tx_outcome"])
            * external_params["dr_amplification_prop_among_nonsuccess"]
        )

        _tb_model.adaptation_functions["dr_amplification"] = dr_amplification_rate
        _tb_model.parameters["dr_amplification"] = "dr_amplification"

    if "age" in STRATIFY_BY:
        age_breakpoints = [0, 5, 15, 60]
        age_infectiousness = get_parameter_dict_from_function(
            logistic_scaling_function(10.0), age_breakpoints
        )
        age_params = get_adapted_age_parameters(age_breakpoints)
        age_params.update(split_age_parameter(age_breakpoints, "contact_rate"))

        # adjustment of latency parameters
        for param in ["early_progression", "late_progression"]:
            for age_break in age_breakpoints:
                if age_break > 5:
                    age_params[param][str(age_break) + "W"] *= external_params[
                        "adult_latency_adjustment"
                    ]

        pop_morts = get_pop_mortality_functions(
            input_database, age_breakpoints, country_iso_code="MNG"
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

        # age-specific IPT
        ipt_by_age = {"ipt_rate": {}}
        for age_break in age_breakpoints:
            ipt_by_age["ipt_rate"][str(age_break)] = external_params[
                "ipt_age_" + str(age_break) + "_ct_coverage"
            ]
        age_params.update(ipt_by_age)

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

        # patch for IPT to overwrite parameters when ds_ipt has been turned off while we still need some coverage at baseline
        if external_params["ds_ipt_switch"] == 0.0 and external_params["mdr_ipt_switch"] == 1.0:
            _tb_model.parameters["ipt_rateXstrain_dsXage_0"] = 0.17
            for age_break in [5, 15, 60]:
                _tb_model.parameters["ipt_rateXstrain_dsXage_" + str(age_break)] = 0.0

    if "organ" in STRATIFY_BY:
        props_smear = {
            "smearpos": external_params["prop_smearpos"],
            "smearneg": 1.0 - (external_params["prop_smearpos"] + 0.20),
            "extrapul": 0.20,
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
        props_location = {
            "rural_province": 0.48,
            "urban_nonger": 0.368,
            "urban_ger": 0.15,
            "prison": 0.002,
        }
        raw_relative_risks_loc = {"rural_province": 1.0}
        for stratum in ["urban_nonger", "urban_ger", "prison"]:
            raw_relative_risks_loc[stratum] = external_params["rr_transmission_" + stratum]
        scaled_relative_risks_loc = scale_relative_risks_for_equivalence(
            props_location, raw_relative_risks_loc
        )

        # dummy matrix for mixing by location
        location_mixing = numpy.array(
            [
                0.899,
                0.05,
                0.05,
                0.001,
                0.049,
                0.7,
                0.25,
                0.001,
                0.049,
                0.25,
                0.7,
                0.001,
                0.1,
                0.1,
                0.1,
                0.7,
            ]
        ).reshape((4, 4))
        location_mixing *= (
            3.0
        )  # adjusted such that heterogeneous mixing yields similar overall burden as homogeneous

        location_adjustments = {}
        for beta_type in ["", "_infected", "_recovered"]:
            location_adjustments["contact_rate" + beta_type] = scaled_relative_risks_loc

        location_adjustments["acf_rate"] = {}
        for stratum in ["rural_province", "urban_nonger", "urban_ger", "prison"]:
            location_adjustments["acf_rate"][stratum] = external_params[
                "acf_" + stratum + "_switch"
            ]

        _tb_model.stratify(
            "location",
            ["rural_province", "urban_nonger", "urban_ger", "prison"],
            [],
            requested_proportions=props_location,
            verbose=False,
            entry_proportions=props_location,
            adjustment_requests=location_adjustments,
            mixing_matrix=location_mixing,
        )

    # _tb_model.transition_flows.to_csv("transitions.csv")
    # _tb_model.death_flows.to_csv("deaths.csv")

    # create some customised derived_outputs

    def notification_function_builder(stratum):
        """
            example of stratum: "Xage_0Xstrain_mdr"
        """

        def calculate_notifications(model, time):

            total_notifications = 0.0
            dict_flows = model.transition_flows.to_dict()

            comp_ind = model.compartment_names.index("infectious" + stratum)
            infectious_pop = model.compartment_values[comp_ind]
            detection_indices = [
                index for index, val in dict_flows["parameter"].items() if "case_detection" in val
            ]
            flow_index = [
                index
                for index in detection_indices
                if dict_flows["origin"][index] == model.compartment_names[comp_ind]
            ][0]
            param_name = dict_flows["parameter"][flow_index]
            detection_tx_rate = model.get_parameter_value(param_name, time)
            tsr = mongolia_tsr(time) + external_params["reduction_negative_tx_outcome"] * (
                1.0 - mongolia_tsr(time)
            )
            if "strain_mdr" in model.compartment_names[comp_ind]:
                tsr = external_params["mdr_tsr"] * external_params["prop_mdr_detected_as_mdr"]
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
            # _tb_model.derived_output_functions['popsize_treatment_support' + stratum] = notification_function_builder(stratum)

    # add output_connections for all stratum-specific incidence outputs
    _tb_model.output_connections.update(
        create_output_connections_for_incidence_by_stratum(_tb_model.compartment_names)
    )

    # prepare death outputs for all strata
    _tb_model.death_output_categories = list_all_strata_for_mortality(_tb_model.compartment_names)

    return _tb_model


def run_model():
    load_model = False
    load_mcmc = False

    scenario_params = {
        # 0: {'contact_rate':1.},
        # 1: {'contact_rate':3.},
        # 2: {'contact_rate':6.},
        # 3: {'contact_rate':10.},
        # 4: {'contact_rate':15.},
        # 5: {'contact_rate':20.},
        # 6: {'contact_rate':25.}
        # 1: {'ipt_age_0_ct_coverage': 1.},
        # 2: {'ipt_age_0_ct_coverage': .5, 'ipt_age_5_ct_coverage': .5, 'ipt_age_15_ct_coverage': .5,
        #          'ipt_age_60_ct_coverage': .5},
        # 3: {'ipt_age_0_ct_coverage': .5, 'ipt_age_5_ct_coverage': .5, 'ipt_age_15_ct_coverage': .5,
        #           'ipt_age_60_ct_coverage': .5, 'ds_ipt_switch': 0., 'mdr_ipt_switch': 1.},
        # 4: {'mdr_tsr': .8},
        # 5: {'reduction_negative_tx_outcome': 0.5},
        # 6: {'acf_coverage': .155, 'acf_urban_ger_switch': 1.},  # 15.5% to get 70,000 screens
        # 7: {'diagnostic_sensitivity_smearneg': 1., 'prop_mdr_detected_as_mdr': .9}
    }
    scenario_list = list(scenario_params.keys())
    if 0 not in scenario_list:
        scenario_list = [0] + scenario_list

    if load_model:
        if load_mcmc:
            models = load_calibration_from_db("mcmc_chistmas_2019", n_burned_per_chain=50)
            scenario_list = [i + 1 for i in range(len(models))]
        else:
            models = []
            scenarios_to_load = scenario_list
            for sc in scenarios_to_load:
                print("Loading model for scenario " + str(sc))
                loaded_model = load_model_scenario(
                    str(sc), database_name="outputs_01_14_2020_10_50_27.db"
                )
                models.append(DummyModel(loaded_model["outputs"], loaded_model["derived_outputs"]))
    else:
        t0 = time()
        models = run_multi_scenario(scenario_params, 2020.0, build_mongolia_model)
        # automatically add combined incidence output
        for model in models:
            outputs_df = pd.DataFrame(model.outputs, columns=model.compartment_names)
            derived_outputs_df = pd.DataFrame(
                model.derived_outputs, columns=model.derived_outputs.keys()
            )
            updated_derived_outputs = add_combined_incidence(derived_outputs_df, outputs_df)
            updated_derived_outputs = updated_derived_outputs.to_dict("list")
            model.derived_outputs = updated_derived_outputs
        store_run_models(models, scenarios=scenario_list, database_name=OUTPUT_DB_PATH)
        delta = time() - t0
        print("Running time: " + str(round(delta, 1)) + " seconds")

    req_outputs = [
        "prevXinfectiousXamong",
        "prevXlatentXamong"
        # 'prevXinfectiousXorgan_smearposXamongXinfectious', 'prevXinfectiousXorgan_smearnegXamongXinfectious',
        # 'prevXinfectiousXorgan_extrapulXamongXinfectious',
        #'prevXinfectiousXorgan_smearposXamongXage_15Xage_60Xlocation_prison']
    ]

    # {'prevXinfectiousXamongXage_15Xage_60': [[2015.], [560.]],
    #                    'prevXlatentXamongXage_5': [[2016.], [9.6]],
    #                    'prevXinfectiousXamongXage_15Xage_60Xhousing_ger': [[2015.], [613.]],
    #                    'prevXinfectiousXamongXage_15Xage_60Xhousing_non-ger': [[2015.], [436.]],
    #                    'prevXinfectiousXamongXage_15Xage_60Xlocation_rural': [[2015.], [529.]],
    #                    'prevXinfectiousXamongXage_15Xage_60Xlocation_province': [[2015.], [513.]],
    #                    'prevXinfectiousXamongXage_15Xage_60Xlocation_urban': [[2015.], [586.]],
    #                    'prevXinfectiousXstrain_mdrXamongXinfectious': [[2016.], [5.3]]
    #                    }

    calib_targets = [
        {
            "output_key": "prevXinfectiousXorgan_smearposXamongXage_15Xage_60",
            "years": [2015.0],
            "values": [204.0],
            "cis": [(143.0, 265.1)],
        },
        {
            "output_key": "prevXinfectiousXorgan_smearnegXamongXage_15Xage_60",
            "years": [2015.0],
            "values": [340.0],
            "cis": [(273.0, 407.0)],
        },
        {
            "output_key": "prevXinfectiousXorgan_smearposXamongXage_15Xage_60Xlocation_rural_province",
            "years": [2015.0],
            "values": [220.0],
        },
        {
            "output_key": "prevXinfectiousXorgan_smearposXamongXage_15Xage_60Xlocation_urban_ger",
            "years": [2015.0],
            "values": [277.0],
        },
        {
            "output_key": "prevXinfectiousXorgan_smearposXamongXage_15Xage_60Xlocation_urban_nonger",
            "years": [2015.0],
            "values": [156],
        },
        {
            "output_key": "prevXinfectiousXamongXage_15Xage_60Xlocation_prison",
            "years": [2015.0],
            "values": [3785],
        },
        {
            "output_key": "prevXlatentXamongXage_5",
            "years": [2016.0],
            "values": [9.6],
            "cis": [(9.02, 10.18)],
        },
        {
            "output_key": "prevXinfectiousXstrain_mdrXamongXinfectious",
            "years": [2015.0],
            "values": [5],
        },
    ]
    calib_targets = []

    targets_to_plot = {}
    for target in calib_targets:
        targets_to_plot[target["output_key"]] = [target["years"], target["values"]]
        if target["output_key"] not in req_outputs:
            req_outputs.append(target["output_key"])

    multipliers = {"prevXinfectiousXstrain_mdrXamongXinfectious": 100.0}

    ymax = {"prevXinfectiousXamong": 2000.0}

    translations = {
        "prevXinfectiousXamong": "TB prevalence (/100,000)",
        "prevXinfectiousXamongXage_0": "TB prevalence among 0-4 y.o. (/100,000)",
        "prevXinfectiousXamongXage_5": "TB prevalence among 5-14 y.o. (/100,000)",
        "prevXinfectiousXamongXage_15": "TB prevalence among 15-59 y.o. (/100,000)",
        "prevXinfectiousXamongXage_60": "TB prevalence among 60+ y.o. (/100,000)",
        "prevXinfectiousXamongXhousing_ger": "TB prev. among Ger population (/100,000)",
        "prevXinfectiousXamongXhousing_non-ger": "TB prev. among non-Ger population(/100,000)",
        "prevXinfectiousXamongXlocation_rural": "TB prev. among rural population (/100,000)",
        "prevXinfectiousXamongXlocation_province": "TB prev. among province population (/100,000)",
        "prevXinfectiousXamongXlocation_urban": "TB prev. among urban population (/100,000)",
        "prevXlatentXamong": "Latent TB infection prevalence (%)",
        "prevXlatentXamongXage_5": "Latent TB infection prevalence among 5-14 y.o. (%)",
        "prevXlatentXamongXage_0": "Latent TB infection prevalence among 0-4 y.o. (%)",
        "prevXinfectiousXamongXage_15Xage_60": "TB prev. among 15+ y.o. (/100,000)",
        "prevXinfectiousXamongXage_15Xage_60Xhousing_ger": "TB prev. among 15+ y.o. Ger population (/100,000)",
        "prevXinfectiousXamongXage_15Xage_60Xhousing_non-ger": "TB prev. among 15+ y.o. non-Ger population (/100,000)",
        "prevXinfectiousXamongXage_15Xage_60Xlocation_rural": "TB prev. among 15+ y.o. rural population (/100,000)",
        "prevXinfectiousXamongXage_15Xage_60Xlocation_province": "TB prev. among 15+ y.o. province population (/100,000)",
        "prevXinfectiousXamongXage_15Xage_60Xlocation_urban": "TB prev. among 15+ y.o. urban population (/100,000)",
        "prevXinfectiousXstrain_mdrXamongXinfectious": "Proportion of MDR-TB among TB (%)",
        "prevXinfectiousXamongXhousing_gerXlocation_urban": "TB prevalence in urban Ger population (/100,000)",
        "age_0": "age 0-4",
        "age_5": "age 5-14",
        "age_15": "age 15-59",
        "age_60": "age 60+",
        "housing_ger": "ger",
        "housing_non-ger": "non-ger",
        "location_rural": "rural",
        "location_province": "province",
        "location_urban": "urban",
        "strain_ds": "DS-TB",
        "strain_mdr": "MDR-TB",
        "incidence": "TB incidence (/100,000/y)",
        "prevXinfectiousXstrain_mdrXamong": "Prevalence of MDR-TB (/100,000)",
    }

    if load_mcmc:
        create_mcmc_outputs(
            models,
            req_outputs=req_outputs,
            out_dir="mcmc_output_plots_test",
            targets_to_plot=targets_to_plot,
            req_multipliers=multipliers,
            translation_dictionary=translations,
            scenario_list=scenario_list,
            ymax=ymax,
            plot_start_time=1990,
        )
    else:
        create_multi_scenario_outputs(
            models,
            req_outputs=req_outputs,
            out_dir="test_inc_stratum",
            targets_to_plot=targets_to_plot,
            req_multipliers=multipliers,
            translation_dictionary=translations,
            scenario_list=scenario_list,
            ymax=ymax,
            plot_start_time=1990,
        )


if __name__ == "__main__":
    run_model()
