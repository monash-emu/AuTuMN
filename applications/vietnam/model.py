import os
from datetime import datetime
from time import time
import yaml

import pandas as pd
from summer_py.summer_model import StratifiedModel


from autumn import constants
from autumn.db import Database
from autumn.tb_model import (
    add_combined_incidence,
    create_output_connections_for_incidence_by_stratum,
    list_all_strata_for_mortality,
    store_run_models,
    get_birth_rate_functions,
)
from autumn.tb_model.latency_params import AGGREGATED_LATENCY_PARAMETERS
from autumn.tool_kit import run_multi_scenario, change_parameter_unit, get_integration_times

from .rate_builder import RateBuilder
from .flows import get_flows
from .stratify.location import stratify_location
from .stratify.age import stratify_age
from .stratify.organ import stratify_organ
from .stratify.strain import stratify_strain
from . import derived_outputs

# Database locations
file_dir = os.path.dirname(os.path.abspath(__file__))
timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
OUTPUT_DB_DIR = os.path.join(constants.DATA_PATH, "vietnam", "databases")
OUTPUT_DB_PATH = os.path.join(OUTPUT_DB_DIR, f"outputs_{timestamp}.db")
PARAMS_PATH = os.path.join(file_dir, "params.yml")
INPUT_DB_PATH = os.path.join(constants.DATA_PATH, "inputs.db")
input_database = Database(database_name=INPUT_DB_PATH)

STRATIFY_BY = ["age", "strain", "location", "organ"]


def build_model(params):
    tb_control_recovery_rate_organ = "smearpos" if "organ" in STRATIFY_BY else "overall"
    rates = RateBuilder(params["rates"], tb_control_recovery_rate_organ)

    flows = get_flows(rates, params["strain"])

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
        initial_conditions=params["population"]["compartments"],
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

    # Apply stratifications.
    if "strain" in STRATIFY_BY:
        strain_params = params["strain"]
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
        age_params = params["age"]
        stratify_age(model, input_database, age_params)

    if "organ" in STRATIFY_BY:
        organ_params = params["organ"]
        stratify_organ(model, organ_params, rates.detect_rate_by_organ)

    if "location" in STRATIFY_BY:
        location_params = params["location"]
        stratify_location(model, location_params)

    # Add 'notification' derviced outputs
    for compartment in model.compartment_names:
        if "infectious" in compartment:
            stratum = compartment.split("infectious")[1]
            func = derived_outputs.build_calc_notifications(stratum, rates, params["strain"])
            model.derived_output_functions[f"notifications{stratum}"] = func

    # Add output_connections for all stratum-specific incidence outputs
    connections = create_output_connections_for_incidence_by_stratum(model.compartment_names)
    model.output_connections.update(connections)

    # Prepare death outputs for all strata
    model.death_output_categories = list_all_strata_for_mortality(model.compartment_names)

    # Calculate population sizes for costing.
    # Get number of detected individuals by strain.
    for tag in ["strain_mdr", "strain_ds", "organ_smearpos", "organ_smearneg", "organ_extrapul"]:
        func = derived_outputs.build_calc_num_detected(tag)
        model.derived_output_functions[f"popsizeXnb_detectedX{tag}"] = func

    # Get active case finding popsize: number of people screened
    pop_size_acf_func = derived_outputs.build_calc_popsize_acf(params["rates"])
    model.derived_output_functions["popsizeXnb_screened_acf"] = pop_size_acf_func

    return model


def run_model():
    with open(PARAMS_PATH, "r") as f:
        params = yaml.safe_load(f)

    if not os.path.exists(OUTPUT_DB_DIR):
        os.makedirs(OUTPUT_DB_DIR, exist_ok=True)

    def _build_model(update_params={}):
        final_params = merge_dicts(update_params, params)
        return build_model(final_params)

    print(f"Starting Vietnam model...")
    start_time = time()
    scenario_params = {}
    models = run_multi_scenario(scenario_params, 2020.0, _build_model)

    # Automatically add combined incidence output
    for model in models:
        outputs_df = pd.DataFrame(model.outputs, columns=model.compartment_names)
        derived_outputs_df = pd.DataFrame(
            model.derived_outputs, columns=model.derived_outputs.keys()
        )
        updated_derived_outputs = add_combined_incidence(derived_outputs_df, outputs_df)
        updated_derived_outputs = updated_derived_outputs.to_dict("list")
        model.derived_outputs = updated_derived_outputs

    scenario_list = list(scenario_params.keys())
    if 0 not in scenario_list:
        scenario_list = [0] + scenario_list

    store_run_models(models, scenarios=scenario_list, database_name=OUTPUT_DB_PATH)
    run_time = time() - start_time
    print(f"Running time: {run_time:0.2f} seconds")


def merge_dicts(src, dest):
    """
    Merge src dict into dest dict.
    """
    for key, value in src.items():
        if isinstance(value, dict):
            # get node or create one
            node = dest.setdefault(key, {})
            merge_dicts(value, node)
        else:
            dest[key] = value

    return dest


if __name__ == "__main__":
    run_model()
