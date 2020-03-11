"""
Build and run any AuTuMN model, storing the outputs
"""

import os
from datetime import datetime
import numpy as np
import pandas as pd
import yaml

from summer_py.constants import IntegrationType
import summer_py.post_processing as post_proc
from autumn.outputs.outputs import Outputs

from autumn.tool_kit.timer import Timer
from autumn.tool_kit import run_multi_scenario
from autumn.tool_kit.utils import make_directory_if_absent, record_parameter_request, record_run_metadata
from autumn.tb_model import add_combined_incidence, store_run_models
from autumn import constants

from applications.marshall_islands.rmi_model import build_rmi_model
from applications.covid_19.covid_model import build_covid_model


FILE_DIR = os.path.dirname(os.path.abspath(__file__))

# Settings for the hand-coded Euler method.
# It's not clear whether this produces reliable results, but is likely faster than odeint
# (~60s at 0.1 step, ~20s at 0.3 step)
EULER_KWARGS = {
    "integration_type": IntegrationType.EULER,
    "solver_args": {"step_size": 0.3},
}
# Settings for the hand-coded Runge-Kutta method, which is more accurate, but slower than Euler.
# It's not clear whether this produces reliable results, but it can be faster than odeint
# (~230s at 0.1 step, ~70s at 0.3 step, ~50s at 0.5 step)
RUNGE_KUTTA_KWARGS = {
    "integration_type": IntegrationType.RUNGE_KUTTA,
    "solver_args": {"step_size": 0.5},
}
# Settings for the SciPy odeint solver - this can get stuck on some ODE types and take a long time (~230s).
ODEINT_KWARGS = {
    "integration_type": IntegrationType.ODE_INT,
}
# ODE solver settings to use when running the model.
SOLVER_KWARGS = ODEINT_KWARGS


def get_post_processing_results(
        models,
        req_outputs,
        req_multipliers,
        outputs_to_plot_by_stratum,
        scenario_list,
        req_times,
        ymax
):

    pps = []
    for scenario_index in range(len(models)):

        # Automatically add some basic outputs
        if hasattr(models[scenario_index], "all_stratifications"):
            for group in models[scenario_index].all_stratifications.keys():
                req_outputs.append("distribution_of_strataX" + group)
                for output in outputs_to_plot_by_stratum:
                    for stratum in models[scenario_index].all_stratifications[group]:
                        req_outputs.append(output + "XamongX" + group + "_" + stratum)

        pps.append(
            post_proc.PostProcessing(
                models[scenario_index],
                requested_outputs=req_outputs,
                scenario_number=scenario_list[scenario_index],
                requested_times=req_times,
                multipliers=req_multipliers,
                ymax=ymax,
            )
        )
        return pps


def run_model(application):

    # Load user information for parameters and outputs from YAML files
    params_path = os.path.join(FILE_DIR, application, 'params.yml')
    outputs_path = os.path.join(FILE_DIR, application, 'outputs.yml')
    with open(params_path, "r") as yaml_file:
        params = yaml.safe_load(yaml_file)
    with open(outputs_path, "r") as yaml_file:
        output_options = yaml.safe_load(yaml_file)

    # Ensure project folder exists
    project_dir = os.path.join(constants.DATA_PATH, application)
    if not os.path.exists(project_dir):
        os.makedirs(project_dir, exist_ok=True)

    # Include user input if requested
    run_name, run_description = 'manual-calibration', ''

    # Create output data folder
    timestamp = datetime.now().strftime("%d-%m-%Y--%H-%M-%S")
    output_directory = os.path.join(project_dir, f"{run_name}-{timestamp}")
    make_directory_if_absent(output_directory, run_name, timestamp)

    # Determine where to save model outputs
    output_db_path = os.path.join(output_directory, "outputs.db")
    plot_path = os.path.join(output_directory, "plots")

    # Save parameter requests and metadata
    record_parameter_request(output_directory, params)
    record_run_metadata(output_directory, run_name, run_description, timestamp)

    # Prepare scenario data
    scenario_params = params["scenarios"]
    scenario_list = [0, *scenario_params.keys()]

    # Run the model
    if application == 'marshall_islands':
        model_function = build_rmi_model
    elif application == 'covid_19':
        model_function = build_covid_model

    with Timer("Running model scenarios"):
        models = run_multi_scenario(
            scenario_params, params["scenario_start"], model_function, run_kwargs=SOLVER_KWARGS,
        )

    # Post-process and save model outputs
    with Timer("Processing model outputs"):
        store_run_models(models, scenarios=scenario_list, database_name=output_db_path)
        if not os.path.exists(plot_path):
            os.mkdir(plot_path)
        pps = get_post_processing_results(
            models,
            output_options['req_outputs'],
            output_options['req_multipliers'],
            output_options['outputs_to_plot_by_stratum'],
            scenario_list,
            {},
            output_options['ymax']
        )

    with Timer('Creating model outputs'):
        outputs = Outputs(
            pps,
            output_options['targets_to_plot'],
            plot_path,
            output_options['translation_dictionary'],
            plot_start_time=output_options['plot_start_time']
        )
        outputs.plot_requested_outputs()
        for output in output_options['outputs_to_plot_by_stratum']:
            for sc_index in range(len(models)):
                outputs.plot_outputs_by_stratum(output, sc_index=sc_index)

        # Plotting the baseline function value, but here in case we want to use for multi-scenario in the future
        for input_function in output_options['functions_to_plot']:
            outputs.plot_input_function(input_function, models[0].adaptation_functions[input_function])


if __name__ == "__main__":
    run_model('covid_19')
