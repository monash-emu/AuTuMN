"""
Build and run the Republic of Marshall Islands (RMI) model, storing the outputs.
"""
import os
from datetime import datetime

import pandas as pd
import yaml
from slugify import slugify

from summer_py.constants import IntegrationType

from autumn.tool_kit.timer import Timer
from autumn.tool_kit import run_multi_scenario
from autumn.tool_kit.utils import get_git_branch, get_git_hash
from autumn.tb_model import add_combined_incidence, store_run_models, create_multi_scenario_outputs
from autumn import constants

from applications.marshall_islands.rmi_model import build_rmi_model


FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PARAMS_PATH = os.path.join(FILE_DIR, "params.yml")
OUTPUTS_PATH = os.path.join(FILE_DIR, "outputs.yml")

# Settings for the hand-coded Euler method.
# It's not clear whether this produces reliable results, but it can be faster than odeint (~60s at 0.1 step, ~20s at 0.3 step)
EULER_KWARGS = {
    "integration_type": IntegrationType.EULER,
    "solver_args": {"step_size": 0.3},
}
# Settings for the hand-coded Runge-Kutta method, which is more accurate, but slower than Euler.
# It's not clear whether this produces reliable results, but it can be faster than odeint (~230s at 0.1 step, ~70s at 0.3 step, ~50s at 0.5 step)
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


def run_model():
    # Load experiment data from YAML files.
    with open(PARAMS_PATH, "r") as f:
        params = yaml.safe_load(f)

    with open(OUTPUTS_PATH, "r") as f:
        output_options = yaml.safe_load(f)

    # Ensure project folder exists.
    project_dir = os.path.join(constants.DATA_PATH, "marshall_islands")
    if not os.path.exists(project_dir):
        os.makedirs(project_dir, exist_ok=True)

    # Get user input.
    experiment_name = slugify(input("Experiment name (empty ok): "))
    experiment_desc = input("Experiment description (empty ok): ")

    # Create experiment folder.
    experiment_name = experiment_name or "experiment"
    timestamp = datetime.now().strftime("%d-%m-%Y--%H-%M-%S")
    experiment_dir = os.path.join(project_dir, f"{experiment_name}-{timestamp}")
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    else:
        raise FileExistsError(f"Experiment {experiment_name} already exists at time {timestamp}.")

    # Figure out where to save model outputs.
    output_db_path = os.path.join(experiment_dir, "outputs.db")
    plot_path = os.path.join(experiment_dir, "plots")

    # Save experiment parameters.
    param_path = os.path.join(experiment_dir, "params.yml")
    with open(param_path, "w") as f:
        yaml.dump(params, f)

    # Save experiment metadata.
    meta_path = os.path.join(experiment_dir, "meta.yml")
    metadata = {
        "name": experiment_name,
        "description": experiment_desc,
        "start_time": timestamp,
        "git_branch": get_git_branch(),
        "git_commit": get_git_hash(),
    }

    with open(meta_path, "w") as f:
        yaml.dump(metadata, f)

    # Prepare scenario data.
    scenario_params = params["scenarios"]
    scenario_list = [0, *scenario_params.keys()]

    # Run the model
    with Timer("Running model scenarios"):
        models = run_multi_scenario(
            scenario_params, params["scenario_start"], build_rmi_model, run_kwargs=SOLVER_KWARGS,
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

    # Save plots of model outputs.
    with Timer("Plotting model outputs"):
        create_multi_scenario_outputs(
            models, out_dir=plot_path, scenario_list=scenario_list, **output_options
        )


if __name__ == "__main__":
    run_model()
