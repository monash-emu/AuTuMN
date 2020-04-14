"""
Build and run any AuTuMN model, storing the outputs
"""
import os
import yaml
from datetime import datetime

from summer_py.constants import IntegrationType
import summer_py.post_processing as post_proc

from autumn.outputs.outputs import (
    OutputPlotter,
    collate_compartment_across_stratification,
    collate_prevalence,
    create_output_dataframes,
    Outputs,
)
from autumn.tool_kit.timer import Timer
from autumn.tool_kit import run_multi_scenario
from autumn.tool_kit.utils import (
    get_git_branch,
    get_git_hash,
)
from autumn.tb_model import store_run_models
from autumn import constants
from autumn.demography.ageing import add_agegroup_breaks


def build_model_runner(
    model_name: str, build_model, params: dict, outputs: dict, mixing_functions=None,
):
    """
    Factory function that returns a 'run_model' function.
    """
    assert model_name, "Value 'model_name' must be set."
    assert build_model, "Value 'build_model' must be set."
    assert params, "Value 'params' must be set."
    assert outputs, "Value 'outputs' must be set."

    def run_model(run_name="model-run", run_description=""):
        """
        Run the model, save the outputs.
        """
        print(f"Running {model_name}...")

        # FIXME: Get rid of rename
        model_function = build_model
        output_options = outputs

        # FIXME: This is model specific, doesn't live here.
        # If agegroup breaks specified in default, add these to the agegroup stratification
        params["default"] = add_agegroup_breaks(params["default"])

        # FIXME: Not clear what this does or why it is here.
        output_options = collate_prevalence(output_options)
        for i_combination in range(len(output_options["output_combinations_to_collate"])):
            output_options = collate_compartment_across_stratification(
                output_options,
                output_options["output_combinations_to_collate"][i_combination][0],
                output_options["output_combinations_to_collate"][i_combination][1],
                params["default"]["all_stratifications"][
                    output_options["output_combinations_to_collate"][i_combination][1]
                ],
            )

        # Ensure project folder exists.
        project_dir = os.path.join(constants.DATA_PATH, model_name)
        if not os.path.exists(project_dir):
            os.makedirs(project_dir, exist_ok=True)

        # Create output data folder.
        timestamp = datetime.now().strftime("%d-%m-%Y--%H-%M-%S")
        output_dir = os.path.join(project_dir, f"{run_name}-{timestamp}")
        if os.path.exists(output_dir):
            raise FileExistsError(f"Experiment {run_name} already exists at time {timestamp}.")
        else:
            os.makedirs(output_dir)

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

        # Prepare scenario data for running.
        scenario_params = params["scenarios"]
        scenario_list = [0, *scenario_params.keys()]

        with Timer("Running model scenarios"):
            models = run_multi_scenario(
                scenario_params,
                params,
                model_function,
                run_kwargs={"integration_type": IntegrationType.ODE_INT},
            )

        # Post-process and save model outputs
        with Timer("Processing model outputs"):
            store_run_models(models, scenarios=scenario_list, database_name=output_db_path)
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)

            pps = []
            for scenario_index in range(len(models)):

                # Automatically add some basic outputs
                if hasattr(models[scenario_index], "all_stratifications"):
                    for group in models[scenario_index].all_stratifications.keys():
                        output_options["req_outputs"].append("distribution_of_strataX" + group)

                pps.append(
                    post_proc.PostProcessing(
                        models[scenario_index],
                        requested_outputs=output_options["req_outputs"],
                        scenario_number=scenario_list[scenario_index],
                        requested_times={},
                        multipliers=output_options["req_multipliers"],
                        ymax=output_options["ymax"],
                    )
                )

            create_output_dataframes(pps, params)

        with Timer("Creating model outputs"):

            # New approach to plotting outputs, intended to be more general
            outputs_plotter = OutputPlotter(models, pps, output_options, output_dir)
            outputs_plotter.save_flows_sheets()
            outputs_plotter.run_input_plots()

            # Old code to plot requested_outputs and derived_outputs
            old_outputs_plotter = Outputs(
                models,
                pps,
                output_options,
                targets_to_plot=output_options["targets_to_plot"],
                out_dir=output_dir,
                plot_start_time=0,
            )
            old_outputs_plotter.plot_requested_outputs()

    return run_model
