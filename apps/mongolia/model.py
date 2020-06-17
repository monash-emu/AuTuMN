import os
from copy import deepcopy

import numpy
from summer.model import (
    StratifiedModel,
    split_age_parameter,
    create_sloping_step_function,
    find_name_components,
    find_stem,
)
from summer.model.utils.parameter_processing import (
    get_parameter_dict_from_function,
    logistic_scaling_function,
)

from autumn import constants
from autumn.curve import scale_up_function
from autumn.db import Database, get_pop_mortality_functions
from autumn.tb_model import (
    create_output_connections_for_incidence_by_stratum,
    list_all_strata_for_mortality,
    scale_relative_risks_for_equivalence,
    provide_aggregated_latency_parameters,
    get_adapted_age_parameters,
    convert_competing_proportion_to_rate,
    add_standard_latency_flows,
    add_standard_natural_history_flows,
    add_standard_infection_flows,
    add_birth_rate_functions,
)
from autumn.tool_kit import return_function_of_function, change_parameter_unit

# Database locations
INPUT_DB_PATH = os.path.join(constants.DATA_PATH, "inputs.db")


def build_model(params: dict) -> StratifiedModel:
    external_params = deepcopy(params)
    model_parameters = {
        "contact_rate": external_params["contact_rate"],
        "contact_rate_recovered": external_params["contact_rate"]
        * external_params["rr_transmission_recovered"],
        "contact_rate_late_latent": external_params["contact_rate"]
        * external_params["rr_transmission_late_latent"],
        "recovery": external_params["self_recovery_rate"],
        "infect_death": external_params["tb_mortality_rate"],
        **external_params,
    }
    stratify_by = external_params["stratify_by"]
    derived_output_types = external_params["derived_outputs"]

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
    compartments = [
        "susceptible",
        "early_latent",
        "late_latent",
        "infectious",
        "recovered",
    ]

    # define model     #replace_deaths  add_crude_birth_rate
    init_pop = {"infectious": 1000, "late_latent": 1000000}

    tb_model = StratifiedModel(
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
    tb_model = add_birth_rate_functions(tb_model, input_database, "MNG")

    # add case detection process to basic model
    tb_model.add_transition_flow(
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
        dict_flows = model.transition_flows_dict
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

    tb_model.add_transition_flow(
        {
            "type": "customised_flows",
            "parameter": "ipt_rate",
            "origin": "early_latent",
            "to": "recovered",
            "function": ipt_flow_func,
        }
    )

    # add ACF flow
    tb_model.add_transition_flow(
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
    if "organ" not in stratify_by:
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
    tb_model.adaptation_functions["case_detection"] = tb_control_recovery_rate
    tb_model.parameters["case_detection"] = "case_detection"

    tb_model.adaptation_functions["ipt_rate"] = ipt_rate_function
    tb_model.parameters["ipt_rate"] = "ipt_rate"

    tb_model.adaptation_functions["acf_rate"] = acf_rate_function
    tb_model.parameters["acf_rate"] = "acf_rate"

    if "strain" in stratify_by:
        mdr_adjustment = (
            external_params["prop_mdr_detected_as_mdr"] * external_params["mdr_tsr"] / 0.9
        )  # /.9 for last DS TSR

        tb_model.stratify(
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

        tb_model.add_transition_flow(
            {
                "type": "standard_flows",
                "parameter": "dr_amplification",
                "origin": "infectiousXstrain_ds",
                "to": "infectiousXstrain_mdr",
                "implement": len(tb_model.all_stratifications),
            }
        )

        dr_amplification_rate = (
            lambda t: detect_rate_by_organ["overall"](t)
            * (1.0 - mongolia_tsr(t))
            * (1.0 - external_params["reduction_negative_tx_outcome"])
            * external_params["dr_amplification_prop_among_nonsuccess"]
        )

        tb_model.adaptation_functions["dr_amplification"] = dr_amplification_rate
        tb_model.parameters["dr_amplification"] = "dr_amplification"

    if "age" in stratify_by:
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
            tb_model.time_variants["universal_death_rateXage_" + str(age_break)] = pop_morts[
                age_break
            ]
            tb_model.parameters[
                "universal_death_rateXage_" + str(age_break)
            ] = "universal_death_rateXage_" + str(age_break)

            age_params["universal_death_rate"][
                str(age_break) + "W"
            ] = "universal_death_rateXage_" + str(age_break)
        tb_model.parameters["universal_death_rateX"] = 0.0

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

        tb_model.stratify(
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
            tb_model.parameters["ipt_rateXstrain_dsXage_0"] = 0.17
            for age_break in [5, 15, 60]:
                tb_model.parameters["ipt_rateXstrain_dsXage_" + str(age_break)] = 0.0

    if "organ" in stratify_by:
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

        tb_model.stratify(
            "organ",
            ["smearpos", "smearneg", "extrapul"],
            ["infectious"],
            infectiousness_adjustments={"smearpos": 1.0, "smearneg": 0.25, "extrapul": 0.0,},
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

    if "location" in stratify_by:
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
        location_mixing *= 3.0  # adjusted such that heterogeneous mixing yields similar overall burden as homogeneous

        location_adjustments = {}
        for beta_type in ["", "_late_latent", "_recovered"]:
            location_adjustments["contact_rate" + beta_type] = scaled_relative_risks_loc

        location_adjustments["acf_rate"] = {}
        for stratum in ["rural_province", "urban_nonger", "urban_ger", "prison"]:
            location_adjustments["acf_rate"][stratum] = external_params[
                "acf_" + stratum + "_switch"
            ]

        tb_model.stratify(
            "location",
            ["rural_province", "urban_nonger", "urban_ger", "prison"],
            [],
            requested_proportions=props_location,
            verbose=False,
            entry_proportions=props_location,
            adjustment_requests=location_adjustments,
            mixing_matrix=location_mixing,
        )

    # tb_model.transition_flows.to_csv("transitions.csv")
    # tb_model.death_flows.to_csv("deaths.csv")

    # create some customised derived_outputs

    if "notifications" in derived_output_types:

        def notification_function_builder(stratum):
            """
                example of stratum: "Xage_0Xstrain_mdr"
            """

            def calculate_notifications(model, time):

                total_notifications = 0.0
                dict_flows = model.transition_flows_dict

                comp_ind = model.compartment_names.index("infectious" + stratum)
                infectious_pop = model.compartment_values[comp_ind]
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

        for compartment in tb_model.compartment_names:
            if "infectious" in compartment:
                stratum = compartment.split("infectious")[1]
                tb_model.derived_output_functions[
                    "notifications" + stratum
                ] = notification_function_builder(stratum)
                # tb_model.derived_output_functions['popsize_treatment_support' + stratum] = notification_function_builder(stratum)

    if "incidence" in derived_output_types:
        # add output_connections for all stratum-specific incidence outputs
        incidence_output_conns = create_output_connections_for_incidence_by_stratum(
            tb_model.compartment_names
        )
        tb_model.output_connections.update(incidence_output_conns)
        # Create a 'combined incidence' derived output
        early_names = [k for k in incidence_output_conns.keys() if k.startswith("incidence_early")]
        for early_name in early_names:
            rootname = early_name[15:]
            late_name = f"incidence_late{rootname}"
            combined_name = f"incidence{rootname}"

            def add_combined_incidence(model, time, e=early_name, l=late_name):
                time_idx = model.times.index(time)
                early_incidence = model.derived_outputs[e][time_idx]
                late_incidence = model.derived_outputs[l][time_idx]
                return early_incidence + late_incidence

            tb_model.derived_output_functions[combined_name] = add_combined_incidence

    if "mortality" in derived_output_types:
        # prepare death outputs for all strata
        tb_model.death_output_categories = list_all_strata_for_mortality(tb_model.compartment_names)

    ############################################
    #       population sizes for costing
    ############################################
    if "popsizes" in derived_output_types:
        # nb of detected individuals by strain:
        def detected_popsize_function_builder(tag):
            """
                example of tag: "starin_mdr" or "organ_smearpos"
            """

            def calculate_nb_detected(model, time):
                nb_treated = 0.0
                for key, value in model.derived_outputs.items():
                    if "notifications" in key and tag in key:
                        this_time_index = model.times.index(time)
                        nb_treated += value[this_time_index]
                return nb_treated

            return calculate_nb_detected

        for tag in [
            "strain_mdr",
            "strain_ds",
            "organ_smearpos",
            "organ_smearneg",
            "organ_extrapul",
        ]:
            tb_model.derived_output_functions[
                "popsizeXnb_detectedX" + tag
            ] = detected_popsize_function_builder(tag)

        # ACF popsize: number of people screened
        def popsize_acf(model, time):
            if external_params["acf_coverage"] == 0.0:
                return 0.0
            pop_urban_ger = sum(
                [
                    model.compartment_values[i]
                    for i, c_name in enumerate(model.compartment_names)
                    if "location_urban_ger" in c_name
                ]
            )
            return external_params["acf_coverage"] * pop_urban_ger

        tb_model.derived_output_functions["popsizeXnb_screened_acf"] = popsize_acf

    return tb_model


def build_mongolia_timevariant_cdr(cdr_multiplier):
    cdr = {
        1950.0: 0.0,
        1980.0: 0.10,
        1990.0: 0.15,
        2000.0: 0.20,
        2010.0: 0.30,
        2015: 0.33,
    }
    return scale_up_function(
        cdr.keys(), [c * cdr_multiplier for c in list(cdr.values())], smoothness=0.2, method=5,
    )


def build_mongolia_timevariant_tsr():
    tsr = {1950.0: 0.0, 1970.0: 0.2, 1994.0: 0.6, 2000.0: 0.85, 2010.0: 0.87, 2016: 0.9}
    return scale_up_function(tsr.keys(), tsr.values(), smoothness=0.2, method=5)
