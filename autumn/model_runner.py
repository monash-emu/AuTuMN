"""
Build and run any AuTuMN model, storing the outputs
"""
import os
import logging
import yaml
from datetime import datetime


from autumn import constants
from autumn.tool_kit.timer import Timer
from autumn.tool_kit.serializer import serialize_model
from autumn.tool_kit.scenarios import Scenario
from autumn.tool_kit.utils import (
    get_git_branch,
    get_git_hash,
)
from autumn.db.models import store_run_models

logger = logging.getLogger(__name__)


def build_model_runner(
    model_name: str, param_set_name: str, build_model, params: dict,
):
    """
    Factory function that returns a 'run_model' function.
    """
    assert model_name, "Value 'model_name' must be set."
    assert build_model, "Value 'build_model' must be set."
    assert params, "Value 'params' must be set."

    if not param_set_name:
        param_set_name = "main-model"

    def run_model(run_scenarios=True):
        """
        Run the model, save the outputs.
        """
        logger.info(f"Running {model_name} {param_set_name}...")

        # Ensure project folder exists.
        project_dir = os.path.join(constants.OUTPUT_DATA_PATH, "run", model_name, param_set_name)
        timestamp = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        output_dir = os.path.join(project_dir, timestamp)
        os.makedirs(output_dir, exist_ok=True)

        # Determine where to save model outputs
        output_db_path = os.path.join(output_dir, "outputs.db")

        # Save model parameters to output dir.
        param_path = os.path.join(output_dir, "params.yml")
        with open(param_path, "w") as f:
            yaml.dump(params, f)

        # Save model run metadata to output dir.
        meta_path = os.path.join(output_dir, "meta.yml")
        metadata = {
            "model_name": model_name,
            "param_set_name": param_set_name,
            "start_time": timestamp,
            "git_branch": get_git_branch(),
            "git_commit": get_git_hash(),
        }
        with open(meta_path, "w") as f:
            yaml.dump(metadata, f)

        with Timer("Running model scenarios"):
            num_scenarios = 1 + len(params["scenarios"].keys())
            scenarios = []
            for scenario_idx in range(num_scenarios):
                scenario = Scenario(build_model, scenario_idx, params)
                scenarios.append(scenario)

            # Run the baseline scenario.
            baseline_scenario = scenarios[0]
            baseline_scenario.run()
            baseline_model = baseline_scenario.model
            save_serialized_model(baseline_model, output_dir, "baseline")

            if not run_scenarios:
                # Do not run non-baseline models
                scenarios = scenarios[:1]

            # Run all the other scenarios
            for scenario in scenarios[1:]:
                scenario.run(base_model=baseline_model)
                name = f"scenario-{scenario.idx}"
                save_serialized_model(scenario.model, output_dir, name)

        with Timer("Saving model outputs to the database"):
            models = [s.model for s in scenarios]
            store_run_models(models, output_db_path)

    return run_model


def save_serialized_model(model, output_dir: str, name: str):
    model_path = os.path.join(output_dir, "models")
    os.makedirs(model_path, exist_ok=True)
    model_filepath = os.path.join(model_path, f"{name}.yml")
    model_data = serialize_model(model)
    with open(model_filepath, "w") as f:
        yaml.dump(model_data, f)
