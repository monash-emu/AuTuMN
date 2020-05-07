import os
import logging
from typing import List, Tuple

import seaborn as sns
import numpy as np
from matplotlib import pyplot
from summer.model.strat_model import StratifiedModel

from autumn.demography.social_mixing import load_specific_prem_sheet
from autumn.tool_kit.scenarios import Scenario
from autumn.tool_kit import schema_builder as sb

from .plotter import Plotter, COLOR_THEME

logger = logging.getLogger(__file__)


# Schema used to validate output plotting configuration data.
validate_plot_config = sb.build_validator(
    # A list of translation mappings used to format plot titles.
    translations=sb.DictGeneric(str, str),
    # List of derived / generated outputs to plot
    outputs_to_plot=sb.List(
        sb.Dict(name=str, target_times=sb.List(float), target_values=sb.List(sb.List(float)))
    ),
    # Plot population distribution across particular strata
    pop_distribution_strata=sb.List(str),
    # Plot prevalence combinations
    prevalence_combos=sb.List(sb.List(str)),
    # Visualise input functions over model time range.
    input_function=sb.Dict(start_time=float, func_names=sb.List(str)),
    # Visualise parameter values across categories for a particular time.
    parameter_category_values=sb.Dict(time=float, param_names=sb.List(str)),
)


def plot_scenarios(scenarios: List[Scenario], out_dir: str, plot_config: dict):
    """
    Plot the model outputs using the supplied config to the output directory.
    """
    validate_plot_config(plot_config)
    translations = plot_config["translations"]
    outputs_to_plot = plot_config["outputs_to_plot"]
    prevalence_combos = plot_config["prevalence_combos"]
    param_config = plot_config["parameter_category_values"]
    input_config = plot_config["input_function"]
    pop_distribution_strata = plot_config["pop_distribution_strata"]

    if len(scenarios) > 1:
        # Create multi-scenario-plots
        multi_out_dir = os.path.join(out_dir, "multi")
        os.makedirs(multi_out_dir, exist_ok=True)
        multi_plotter = Plotter(multi_out_dir, translations)
        for output_config in outputs_to_plot:
            plot_outputs_multi(multi_plotter, scenarios, output_config)

    # Create scenario-specifc plots
    for scenario in scenarios:
        model_out_dir = os.path.join(out_dir, scenario.name)
        os.makedirs(model_out_dir, exist_ok=True)
        scenario_plotter = Plotter(model_out_dir, translations)
        model = scenario.model
        generated_outputs = scenario.generated_outputs
        for output_config in outputs_to_plot:
            plot_outputs_single(scenario_plotter, scenario, output_config)
        plot_mixing_matrix(scenario_plotter, model)
        plot_prevalence_combinations(scenario_plotter, model, prevalence_combos, generated_outputs)
        if scenario.is_baseline:
            # Only plot some graphs for the base model.
            plot_parameter_category_values(
                scenario_plotter, model, param_config["param_names"], param_config["time"]
            )
            plot_input_function(
                scenario_plotter, model, input_config["func_names"], input_config["start_time"]
            )
            plot_pop_distribution_by_stratum(
                scenario_plotter, model, pop_distribution_strata, generated_outputs
            )
            plot_exponential_growth_rate(scenario_plotter, model)


def plot_agg_compartments_multi_scenario(
    plotter: Plotter, scenarios: List[Scenario], compartment_names: List[str], is_logscale=False
):
    """
    Plot multiple compartments with values aggregated for a multiple scenarios.
    """
    fig, axis, _, _, _ = plotter.get_figure()
    legend = []
    for color_idx, scenario in enumerate(scenarios):
        model = scenario.model
        values = np.zeros(model.outputs.shape[0])
        for compartment_name in compartment_names:
            comp_idx = model.compartment_names.index(compartment_name)
            values += model.outputs[:, comp_idx]

        axis.plot(model.times, values, color=COLOR_THEME[color_idx], alpha=0.7)
        legend.append(scenario.name)

    axis.legend(legend)
    if is_logscale:
        axis.set_yscale("log")

    plotter.save_figure(fig, filename="aggregate compartments", title_text="aggregate compartments")


def plot_single_compartment_multi_scenario(
    plotter: Plotter, scenarios: List[Scenario], compartment_name: str, is_logscale=False
):
    """
    Plot the selected output compartment for a multiple scenarios.
    """
    fig, axis, _, _, _ = plotter.get_figure()
    legend = []
    for color_idx, scenario in enumerate(scenarios):
        model = scenario.model
        comp_idx = model.compartment_names.index(compartment_name)
        values = model.outputs[:, comp_idx]
        axis.plot(model.times, values, color=COLOR_THEME[color_idx], alpha=0.7)
        legend.append(scenario.name)

    axis.legend(legend)
    if is_logscale:
        axis.set_yscale("log")

    plotter.save_figure(fig, filename=compartment_name, title_text=compartment_name)


def plot_multi_compartments_single_scenario(
    plotter: Plotter, scenario: Scenario, compartments: List[str], is_logscale=False
):
    """
    Plot the selected output compartments for a single scenario.
    """
    model = scenario.model
    times = model.times

    fig, axis, _, _, _ = plotter.get_figure()
    legend = []
    for color_idx, compartment_name in enumerate(reversed(compartments)):
        comp_idx = model.compartment_names.index(compartment_name)
        values = model.outputs[:, comp_idx]
        axis.plot(times, values, color=COLOR_THEME[color_idx], alpha=0.7)
        legend.append(compartment_name)

    axis.legend(legend)
    if is_logscale:
        axis.set_yscale("log")

    plotter.save_figure(fig, filename="compartments", title_text="compartments")


def plot_outputs_multi(
    plotter: Plotter, scenarios: List[Scenario], output_config: dict, is_logscale=False
):
    """
    Plot the model derived/generated outputs requested by the user for multiple single scenarios, on one plot.
    """
    fig, axis, _, _, _ = plotter.get_figure()
    output_name = output_config["name"]
    legend = []
    for idx, scenario in enumerate(reversed(scenarios)):
        color_idx = len(scenarios) - idx - 1
        _plot_outputs_to_axis(axis, scenario, output_name, color_idx=color_idx, alpha=0.7)
        legend.append(scenario.name)

    axis.legend(legend)
    target_values = output_config["target_values"]
    target_times = output_config["target_times"]
    _plot_targets_to_axis(axis, target_values, target_times)
    if is_logscale:
        axis.set_yscale("log")

    plotter.save_figure(fig, filename=output_name, title_text=output_name)


def plot_outputs_single(plotter: Plotter, scenario: Scenario, output_config: dict):
    """
    Plot the model derived/generated outputs requested by the user for a single scenario.
    """
    fig, axis, _, _, _ = plotter.get_figure()
    output_name = output_config["name"]
    target_values = output_config["target_values"]
    target_times = output_config["target_times"]
    _plot_outputs_to_axis(axis, scenario, output_name)
    _plot_targets_to_axis(axis, target_values, target_times)
    plotter.save_figure(fig, filename=output_name, subdir="outputs", title_text=output_name)


def _plot_outputs_to_axis(axis, scenario: Scenario, name: str, color_idx=0, alpha=1):
    """
    Plot outputs requested by output_config from scenario to the provided axis.
    """
    model = scenario.model
    generated_outputs = scenario.generated_outputs
    if generated_outputs and name in generated_outputs:
        plot_values = generated_outputs[name]
    elif name in model.derived_outputs:
        plot_values = model.derived_outputs[name]
    else:
        logger.error("Could not plot output named %s - not found.", name)
        return

    # Plot the values as a line.
    if type(plot_values) is list:
        axis.plot(model.times, plot_values, color=COLOR_THEME[color_idx], alpha=alpha)
    else:
        logger.error("Could not plot output named %s - non-list data format.", name)


def _plot_targets_to_axis(axis, target_values: List[float], target_times: List[int]):
    """
    Plot output value calibration targets as points on the axis.
    """
    assert len(target_times) == len(target_values), "Targets have inconsistent length"
    for i, time in enumerate(target_times):
        values = target_values[i]
        is_confidence_interval = len(values) > 1
        if is_confidence_interval:
            # Plot confidence interval
            x_vals = [time, time]
            y_vals = values[1:]
            axis.plot(x_vals, y_vals, "m", linewidth=1, color="red")
        else:
            # Plot a single point estimate
            value = values[0]
            marker_size = 30.0
            axis.scatter(time, value, marker="o", color="red", s=30)
            axis.scatter(time, value, marker="o", color="white", s=10)


def plot_exponential_growth_rate(plotter: Plotter, model: StratifiedModel):
    """
    Find the exponential growth rate at a range of requested time points
    """
    logger.error("plot_exponential_growth_rate does not work yet")
    return
    growth_rates = []
    for time_idx in range(len(model.times) - 1):
        try:
            incidence_values = model.derived_outputs["incidence"]
        except KeyError:
            logger.error("No derived output called 'incidence'")
            return

        start_idx, end_idx = time_idx, time_idx + 1
        start_time, end_time = model.times[start_idx], model.times[end_idx]
        start_val, end_val = incidence_values[start_idx], incidence_values[end_idx]
        exponential_growth_rate = np.log(end_val / start_val) / (start_time - start_time)
        growth_rates.append(exponential_growth_rate)

    fig, axis, max_dims, n_rows, n_cols = plotter.get_figure()
    axis.plot(model.times[:-1], growth_rates)
    axis.set_ylim([0.0, 0.25])
    filename = "exponential_growth_rate"
    plotter.save_figure(fig, filename=filename, title_text=filename)


# FIXME: Assumes particular generated strata are present.
def plot_pop_distribution_by_stratum(
    plotter: Plotter, model: StratifiedModel, strata: List[str], generated_outputs: dict
):
    for strata_name in strata:
        fig, axes, max_dims, n_rows, n_cols = plotter.get_figure()
        previous_values = [0.0] * len(model.times)
        for stratum_idx, stratum in enumerate(model.all_stratifications[strata_name]):
            colour = stratum_idx / len(model.all_stratifications[strata_name])
            key = f"distribution_of_strataX{strata_name}"
            try:
                working_values = generated_outputs[key][stratum]
            except KeyError:
                logger.error("No generated output found for %s", key)
                continue

            new_values = [
                working + previous for working, previous in zip(working_values, previous_values)
            ]
            axes.fill_between(
                model.times, previous_values, new_values, color=(colour, 0.0, 1 - colour),
            )
            previous_values = new_values

        axes.legend(model.all_stratifications[strata_name])
        plotter.save_figure(fig, filename=f"distribution_by_stratum_{strata_name}")


def plot_prevalence_combinations(
    plotter: Plotter,
    model: StratifiedModel,
    prevalence_combos: List[Tuple[str, str]],
    generated_outputs: dict,
):
    """
    Create output graph for each requested stratification, displaying prevalence of compartment in each stratum
    """
    fig, axes, max_dims, n_rows, n_cols = plotter.get_figure()
    for compartment, stratum in prevalence_combos:
        strata_to_iterate = model.all_stratifications[stratum]
        plot_name = f"prevX{compartment}XamongX{stratum}"
        for stratum_idx, stratum in enumerate(strata_to_iterate):
            colour = stratum_idx / len(strata_to_iterate)
            values = generated_outputs[plot_name]
            axes.plot(model.times, values, color=(colour, 0.0, 1.0 - colour))

        axes.legend(strata_to_iterate)
        plotter.save_figure(fig, filename=plot_name, title_text=plot_name)


def plot_input_function(
    plotter: Plotter, model: StratifiedModel, func_names: List[str], plot_start_time: float
):
    """
    Plot single simple plot of a function over time
    """
    times = model.times
    for func_name in func_names:
        # Plot requested func names.
        fig, axes, max_dims, n_rows, n_cols = plotter.get_figure()
        colour_index = 0
        param_names = []
        for param_name, param_func in model.final_parameter_functions.items():
            if param_name.startswith(func_name):
                # Plot all parameter functions starting with requested func name.
                colour_index += 1
                param_names.append(param_name)
                values = list(map(param_func, times))
                axes.plot(times, values, color=COLOR_THEME[colour_index])

        axes.legend(param_names)
        plotter.tidy_x_axis(
            axes, start=plot_start_time, end=max(times), max_dims=max_dims, x_label="time",
        )
        plotter.tidy_y_axis(axes, quantity="", max_dims=max_dims)
        plotter.save_figure(fig, filename=func_name, title_text=func_name)


def plot_parameter_category_values(
    plotter: Plotter, model: StratifiedModel, param_names: List[str], plot_time: float,
):
    """
    Create a plot to visualise:
        - the parameter values; and
        - the parameter values across categories after stratification adjustments have been applied
    """
    # Loop over each requested parameter name.
    for param_name in param_names:

        # Collate all the individual parameter names that need to be plotted.
        plot_keys = []
        plot_values = []
        for p_name, p_func in model.final_parameter_functions.items():
            if p_name.startswith(param_name):
                plot_keys.append(p_name)
                plot_values.append(p_func(plot_time))

        # Plot the parameter values
        fig, axis, max_dims, n_rows, n_cols = plotter.get_figure()
        axis.plot(plot_values, linewidth=0.0, marker="o", markersize=5)

        # Labelling of the x-ticks with the parameter names
        pyplot.xticks(list(range(len(plot_keys))))
        x_tick_labels = [key[len(param_name) + 1 :] for key in plot_keys]
        axis.set_xticklabels(x_tick_labels, rotation=90, fontsize=5)

        plotter.save_figure(fig, filename=param_name, title_text=param_name)


def plot_mixing_matrix(plotter: Plotter, model: StratifiedModel):
    """
    Plot the model's mixing matrix
    """
    # Disable code for now:
    # - it is hard coded for Australia (it's not always for Austalia)
    # - it has hard coded locations (locations can change)
    # - it assumes the mixing matrix is location based (can be age groups)
    logger.error("Mixing matrix plotter does not work - no matrices will be plotted.")
    return

    if model.mixing_matrix is None:
        logger.debug("No mixing matrix found, skipping model.")
        return

    fig, axis, max_dims, n_rows, n_cols = plotter.get_figure()

    axis = sns.heatmap(
        model.mixing_matrix,
        yticklabels=model.mixing_categories,
        xticklabels=False,
        vmin=0.0,
        vmax=12.0,
    )
    plotter.save_figure(fig, "mixing_matrix")

    for location in ["all_locations", "school", "home", "work", "other_locations"]:
        fig, axis, max_dims, n_rows, n_cols = plotter.get_figure()
        axis = sns.heatmap(
            load_specific_prem_sheet(location, "Australia"),
            yticklabels=model.mixing_categories,
            xticklabels=False,
            vmin=0.0,
            vmax=12.0,
        )
        plotter.save_figure(fig, f"mixing_matrix_{location}")
