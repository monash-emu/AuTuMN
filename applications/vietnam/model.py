import os
from datetime import datetime
from time import time
from copy import deepcopy
import yaml

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
    get_birth_rate_functions,
    create_multi_scenario_outputs,
    create_mcmc_outputs,
    DummyModel,
)
from autumn.tb_model.latency_params import AGGREGATED_LATENCY_PARAMETERS
from autumn.tool_kit import (
    run_multi_scenario,
    return_function_of_function,
    change_parameter_unit,
    get_integration_times,
)

from .rate_builder import RateBuilder
from .flows import get_flows
from .stratify.location import stratify_location
from .stratify.age import stratify_age
from .stratify.organ import stratify_organ
from .stratify.strain import stratify_strain

# Database locations
file_dir = os.path.dirname(os.path.abspath(__file__))
timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
INPUT_DB_PATH = os.path.join(constants.DATA_PATH, "inputs.db")
input_database = Database(database_name=INPUT_DB_PATH)
OUTPUT_DB_PATH = os.path.join(
    constants.DATA_PATH, "vietnam", "databases", f"outputs_{timestamp}.db"
)
PARAMS_PATH = os.path.join(file_dir, "params.yml")

STRATIFY_BY = ["age", "strain", "location", "organ"]


def build_model(params):
    tb_control_recovery_rate_organ = "smearpos" if "organ" in STRATIFY_BY else "overall"
    rates = RateBuilder(params['rates'], tb_control_recovery_rate_organ)

    flows = get_flows() # FIXME: Add rate func to args

    start, end, step = params["time"]["start"], params["time"]["end"], params["time"]["step"]
    times = get_integration_times(start, end, step)

    contact_rate = params["model"]["contact_rate"]
    contact_rate_recovered = contact_rate * params["model"]["rr_transmission_recovered"]
    contact_rate_infected = contact_rate * params["model"]["rr_transmission_infected"]
    latency_params = change_parameter_unit(AGGREGATED_LATENCY_PARAMETERS, 365.251)
    model_params = {
        **params["model"],
        "contact_rate_recovered": contact_rate_recovered,
        "contact_rate_infected": contact_rate_infected,
        **latency_params,
    }
    model = StratifiedModel(
        times=times,
        compartment_types=params["compartments"],
        initial_conditions=params['population']['compartments']
        parameters=model_params,
        requested_flows=flows,
        birth_approach="replace_deaths",
        starting_population=params["population"]["total"],
        death_output_categories=((), ("age_0",)),
    )

    # Add crude birth rate from UN estimates.
    model = get_birth_rate_functions(model, input_database, "VNM")

    # Assign time-varying rate functions to model parameters
    model.adaptation_functions["case_detection"] = rates.get_tb_control_recovery
    model.parameters["case_detection"] = "case_detection"

    model.adaptation_functions["ipt_rate"] = rates.get_isoniazid_preventative_therapy
    model.parameters["ipt_rate"] = "ipt_rate"

    model.adaptation_functions["acf_rate"] = rates.get_active_case_finding
    model.parameters["acf_rate"] = "acf_rate"

    if "strain" in STRATIFY_BY:
        strain_params = params['strain']
        stratify_strain(model, strain_params)
        flow = {
            "type": "standard_flows",
            "parameter": "dr_amplification",
            "origin": "infectiousXstrain_ds",
            "to": "infectiousXstrain_mdr",
            "implement": len(model.all_stratifications),  
        }
        model.add_transition_flow(flow)
        model.adaptation_functions["dr_amplification"] = rates.get_dr_amplification
        model.parameters["dr_amplification"] = "dr_amplification"

    if "age" in STRATIFY_BY:
        age_params = params['age']
        stratify_age(model, input_database, age_params)

    if "organ" in STRATIFY_BY:
        organ_params = params['organ']
        stratify_organ(model, organ_params, detect_rate_by_organ)

    if "location" in STRATIFY_BY:
        location_params = params['location']
        stratify_location(model, location_params)


    # create some customised derived_outputs

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

    for compartment in model.compartment_names:
        if "infectious" in compartment:
            stratum = compartment.split("infectious")[1]
            model.derived_output_functions[
                "notifications" + stratum
            ] = notification_function_builder(stratum)
            # model.derived_output_functions['popsize_treatment_support' + stratum] = notification_function_builder(stratum)

    # add output_connections for all stratum-specific incidence outputs
    model.output_connections.update(
        create_output_connections_for_incidence_by_stratum(model.compartment_names)
    )

    # prepare death outputs for all strata
    model.death_output_categories = list_all_strata_for_mortality(model.compartment_names)

    ############################################
    #       population sizes for costing
    ############################################

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

    for tag in ["strain_mdr", "strain_ds", "organ_smearpos", "organ_smearneg", "organ_extrapul"]:
        model.derived_output_functions[
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

    model.derived_output_functions["popsizeXnb_screened_acf"] = popsize_acf

    return model


def run_model():
    with open(PARAMS_PATH, "r") as f:
        params = yaml.safe_load(f)

    load_model = True
    load_mcmc = True

    scenario_params = {
        # 1: {'ipt_age_0_ct_coverage': 1.},
        # 2: {'ipt_age_0_ct_coverage': .5, 'ipt_age_5_ct_coverage': .5, 'ipt_age_15_ct_coverage': .5,
        #          'ipt_age_60_ct_coverage': .5},
        # 3: {'ipt_age_0_ct_coverage': .5, 'ipt_age_5_ct_coverage': .5, 'ipt_age_15_ct_coverage': .5,
        #           'ipt_age_60_ct_coverage': .5, 'ds_ipt_switch': 0., 'mdr_ipt_switch': 1.},
        # 4: {'mdr_tsr': .8},
        # 5: {'reduction_negative_tx_outcome': 0.5},
        # 6: {'acf_coverage': .155, 'acf_urban_ger_switch': 1.},  # 15.5% to get 70,000 screens
        # 7: {'diagnostic_sensitivity_smearneg': 1., 'prop_mdr_detected_as_mdr': .9},
        # 8: {'ipt_age_0_ct_coverage': .5, 'ipt_age_5_ct_coverage': .5, 'ipt_age_15_ct_coverage': .5,
        #     'ipt_age_60_ct_coverage': .5, 'ds_ipt_switch': 0., 'mdr_ipt_switch': 1.,
        #     'mdr_tsr': .8,
        #     'reduction_negative_tx_outcome': 0.5,
        #     'acf_coverage': .155, 'acf_urban_ger_switch': 1.,
        #     'diagnostic_sensitivity_smearneg': 1., 'prop_mdr_detected_as_mdr': .9
        #     },
        # 9: {'contact_rate': 0.}
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

    targets_to_plot = {
        "prevXinfectiousXamong": {"times": [2015], "values": [[757.0, 620.0, 894.0]]},
        "prevXlatentXamongXage_5": {"times": [2016], "values": [[9.60, 9.02, 10.18]]},
        "prevXinfectiousXstrain_mdrXamongXinfectious": {
            "times": [2015],
            "values": [[5.03, 4.10, 6.70]],
        },
        "notifications": {
            "times": list(numpy.linspace(1990, 2018, 29)),
            "values": [
                [1659],
                [1611],
                [1516],
                [1418],
                [1730],
                [2780],
                [4062],
                [3592],
                [2915],
                [3348],
                [3109],
                [3526],
                [3829],
                [3918],
                [4542],
                [4601],
                [5049],
                [4654],
                [4490],
                [4481],
                [4458],
                [4217],
                [4128],
                [4331],
                [4483],
                [4685],
                [4425],
                [4220],
                [4065],
            ],
        },
    }

    for target in targets_to_plot.keys():
        if target not in req_outputs and target[0:5] == "prevX":
            req_outputs.append(target)

    multipliers = {"prevXinfectiousXstrain_mdrXamongXinfectious": 100.0}

    ymax = {"prevXinfectiousXamong": 2000.0, "prevXlatentXamongXage_5": 20.0}

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
            out_dir="mcmc_output_12_02",
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
            out_dir="test_new_scenarios",
            targets_to_plot=targets_to_plot,
            req_multipliers=multipliers,
            translation_dictionary=translations,
            scenario_list=scenario_list,
            ymax=ymax,
            plot_start_time=1990,
        )


if __name__ == "__main__":
    run_model()
