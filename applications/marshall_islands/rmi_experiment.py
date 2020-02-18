"""
Build and run the Republic of Marshall Islands (RMI) model, storing the outputs.
"""
import os
from datetime import datetime

import pandas as pd
import yaml
from slugify import slugify

from autumn.tool_kit.timer import Timer
from autumn.tool_kit import run_multi_scenario
from autumn.tb_model import add_combined_incidence, store_run_models, create_multi_scenario_outputs
from autumn import constants

# This is a hack to get the imports to work in PyCharm and in the automated tests.
try:
    # Try import for PyCharm, as if this were a script.
    from rmi_model import build_rmi_model
except ModuleNotFoundError:
    # Try import as if we are in a module.
    from .rmi_model import build_rmi_model


FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PARAMS_PATH = os.path.join(FILE_DIR, "params.yml")
OUTPUTS_PATH = os.path.join(FILE_DIR, "outputs.yml")


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

    # Save experiment metadata.
    param_path = os.path.join(experiment_dir, "params.yml")
    meta_path = os.path.join(experiment_dir, "meta.yml")
    metadata = get_experiment_metadata(experiment_name)
    with open(meta_path, "w") as f:
        yaml.dump(metadata, f)

    with open(param_path, "w") as f:
        yaml.dump(params, f)

    # Prepare scenario data.
    scenario_params = params["scenarios"]
    scenario_list = [0, *scenario_params.keys()]

    # Run the model
    with Timer("Running model scenarios"):
        models = run_multi_scenario(scenario_params, params["scenario_start"], build_rmi_model)

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


def get_experiment_metadata(experiment_name):
    """
    Get experiment metadata for future reference.
    TODO: git commit, datetime, runtime(s), description.
    """
    return {
        "experiment_name": experiment_name,
    }


if __name__ == "__main__":
    run_model()
