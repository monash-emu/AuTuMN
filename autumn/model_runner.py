"""
Build and run any AuTuMN model, storing the outputs

TODO: Add ability to re-load models and plot their results
TODO: Add ability to load MCMC outputs and plot their results

"""
import os
import yaml
from datetime import datetime

from summer_py.post_processing.processor import post_process, validate_post_process_config

from autumn import constants
from autumn.outputs import save_flows_sheets, plot_scenarios, validate_plot_config
from autumn.tool_kit.timer import Timer
from autumn.tool_kit.scenarios import Scenario
from autumn.tool_kit.utils import (
    get_git_branch,
    get_git_hash,
)
from autumn.tb_model import store_run_models


def build_model_runner(
    model_name: str, build_model, params: dict, post_processing_config={}, plots_config={}
):
    """
    Factory function that returns a 'run_model' function.
    """
    assert model_name, "Value 'model_name' must be set."
    assert build_model, "Value 'build_model' must be set."
    assert params, "Value 'params' must be set."

    def run_model(run_name="model-run", run_description=""):
        """
        Run the model, save the outputs.
        """
        print(f"Running {model_name}...")
        if post_processing_config:
            validate_post_process_config(post_processing_config)

        if plots_config:
            validate_plot_config(plots_config)

        # Ensure project folder exists.
        project_dir = os.path.join(constants.DATA_PATH, model_name)
        timestamp = datetime.now().strftime("%d-%m-%Y--%H-%M-%S")
        output_dir = os.path.join(project_dir, f"{run_name}-{timestamp}")
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
            "name": run_name,
            "description": run_description,
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

            # Run all the other scenarios
            for scenario in scenarios[1:]:
                scenario.run(base_model=baseline_model)

        with Timer("Saving model outputs to the database"):
            models = [s.model for s in scenarios]
            scenario_idxs = range(num_scenarios)
            store_run_models(models, scenarios=scenario_idxs, database_name=output_db_path)

        if post_processing_config:
            with Timer("Applying post-processing to model outputs"):
                # Calculate generated outputs with post-processing.
                for scenario in scenarios:
                    scenario.generated_outputs = post_process(
                        scenario.model, post_processing_config
                    )

        if plots_config:
            with Timer("Creating plots"):
                # Plot all scenario outputs.
                plot_dir = os.path.join(output_dir, "plots")
                os.makedirs(plot_dir, exist_ok=True)
                plot_scenarios(scenarios, plot_dir, plots_config)

                # Save some CSV sheets describing the baseline model.
                save_flows_sheets(baseline_model, output_dir)

    return run_model
