"""
Build and run any AuTuMN model, storing the outputs
"""

import os
from datetime import datetime
import pandas as pd
import yaml

from summer_py.constants import IntegrationType

from autumn.tool_kit.timer import Timer
from autumn.tool_kit import run_multi_scenario
from autumn.tool_kit.utils import make_directory_if_absent, record_parameter_request, record_run_metadata
from autumn.tb_model import add_combined_incidence, store_run_models, create_multi_scenario_outputs
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


def run_model(application):
    # Load user information for parameters and outputs from YAML files
    params_path = os.path.join(FILE_DIR, application, "params.yml")
    outputs_path = os.path.join(FILE_DIR, application, "outputs.yml")

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
    with Timer("Post processing model outputs"):
        # Automatically add combined incidence output
        for model in models:
            outputs_df = pd.DataFrame(model.outputs, columns=model.compartment_names)
            derived_outputs_df = pd.DataFrame(
                model.derived_outputs, columns=model.derived_outputs.keys()
            )
            updated_derived_outputs = add_combined_incidence(derived_outputs_df, outputs_df)
            updated_derived_outputs = updated_derived_outputs.to_dict("list")
            model.derived_outputs = updated_derived_outputs

        store_run_models(models, scenarios=scenario_list, database_name=output_db_path)

    # Save plots of model outputs
    with Timer("Plotting model outputs"):
        create_multi_scenario_outputs(
            models, out_dir=plot_path, scenario_list=scenario_list, **output_options,
            input_functions_to_plot=["case_detection"]
        )


if __name__ == "__main__":
    run_model('marshall_islands')
