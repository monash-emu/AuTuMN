import os
from typing import List

from autumn.tool_kit import Scenario

from . import plots
from .plotter import FilePlotter


def plot_scenarios(scenarios: List[Scenario], out_dir: str, plot_config: dict):
    """
    Plot the model outputs using the supplied config to the output directory.
    """
    plots.validate_plot_config(plot_config)
    translations = plot_config["translations"]
    outputs_to_plot = plot_config["outputs_to_plot"]

    if len(scenarios) > 1:
        # Create multi-scenario-plots
        multi_out_dir = os.path.join(out_dir, "multi")
        os.makedirs(multi_out_dir, exist_ok=True)
        multi_plotter = FilePlotter(multi_out_dir, translations)
        for output_config in outputs_to_plot:
            plots.plot_outputs_multi(multi_plotter, scenarios, output_config)

    # Create scenario-specifc plots
    for scenario in scenarios:
        model_out_dir = os.path.join(out_dir, scenario.name)
        os.makedirs(model_out_dir, exist_ok=True)
        scenario_plotter = FilePlotter(model_out_dir, translations)
        model = scenario.model
        generated_outputs = scenario.generated_outputs
        for output_config in outputs_to_plot:
            plots.plot_outputs_single(scenario_plotter, scenario, output_config)
