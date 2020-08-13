"""
Different types of plots that use a Plotter
"""
import logging
from typing import List, Tuple, Callable
from random import choices
import os
import copy

import pandas as pd
import seaborn as sns
import numpy as np
from math import log
from matplotlib import pyplot
from summer.model.strat_model import StratifiedModel

from autumn.inputs import get_country_mixing_matrix
from autumn.tool_kit.scenarios import Scenario
from autumn.tool_kit import schema_builder as sb
from autumn.tool_kit.uncertainty import export_mcmc_quantiles
from autumn.db.database import Database
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches


from .plotter import Plotter, COLOR_THEME
import yaml

logger = logging.getLogger(__name__)


# Schema used to validate output plotting configuration data.
validate_plot_config = sb.build_validator(
    # A list of translation mappings used to format plot titles.
    translations=sb.DictGeneric(str, str),
    # List of derived / generated outputs to plot
    outputs_to_plot=sb.List(
        sb.Dict(name=str, target_times=sb.List(float), target_values=sb.List(sb.List(float)))
    ),
)


def plot_mcmc_parameter_trace(plotter: Plotter, mcmc_tables: List[pd.DataFrame], param_name: str):
    """
    Plot the prameter traces for each MCMC run.
    """
    _overwrite_non_accepted_mcmc_runs(mcmc_tables, column_name=param_name)
    fig, axis, _, _, _ = plotter.get_figure()
    for idx, table_df in enumerate(mcmc_tables):
        table_df[param_name].plot.line(ax=axis, alpha=0.8, linewidth=0.7)

    axis.set_ylabel(param_name)
    axis.set_xlabel("MCMC iterations")
    plotter.save_figure(fig, filename=f"{param_name}-traces", title_text=f"{param_name}-traces")


def plot_loglikelihood_trace(plotter: Plotter, mcmc_tables: List[pd.DataFrame], burn_in=0):
    """
    Plot the loglikelihood traces for each MCMC run.
    """
    _overwrite_non_accepted_mcmc_runs(mcmc_tables, column_name="loglikelihood")
    fig, axis, _, _, _ = plotter.get_figure()

    for idx, table_df in enumerate(mcmc_tables):
        table_df.loglikelihood.plot.line(ax=axis, alpha=0.8, linewidth=0.7)

    axis.set_ylabel("Loglikelihood")
    axis.set_xlabel("MCMC iterations")

    if burn_in:
        axis.axvline(x=burn_in, color=COLOR_THEME[1], linestyle="dotted")
        y_min = min(table_df.loglikelihood[burn_in:])
        y_max = max(table_df.loglikelihood[burn_in:])
        axis.set_ylim((y_min - 0.2 * (y_max - y_min), y_max + 0.2 * (y_max - y_min)))

    plotter.save_figure(fig, filename="loglikelihood-traces", title_text="loglikelihood-traces")


def plot_burn_in(plotter: Plotter, num_iters: int, burn_in: int):
    """
    Plot the trade off been num iters and burn-in for MCMC runs.
    """
    fig, axis, _, _, _ = plotter.get_figure()

    def floor(n):
        val = num_iters - n
        return val if val > 0 else 0

    values = [floor(i) for i in range(num_iters)]

    fig, axis, _, _, _ = plotter.get_figure()

    axis.plot(values, color=COLOR_THEME[0])
    axis.set_ylabel("Number iters after burn-in")
    axis.set_xlabel("Burn-in size")
    axis.set_ylim(bottom=-5, top=num_iters)
    axis.set_xlim(left=0, right=num_iters)
    axis.axvline(x=burn_in, color=COLOR_THEME[1], linestyle="dotted")
    axis.axhline(y=num_iters - burn_in, color=COLOR_THEME[1], linestyle="dotted")
    plotter.save_figure(fig, filename="burn-in", title_text="burn-in")


def plot_posterior(
    plotter: Plotter, mcmc_tables: List[pd.DataFrame], param_name: str, num_bins: int
):
    """
    Plots the posterior distribution of a given parameter in a histogram.
    """
    _overwrite_non_accepted_mcmc_runs(mcmc_tables, column_name=param_name)
    vals = mcmc_tables[0][param_name]
    for table_df in mcmc_tables[1:]:
        vals.append(table_df[param_name])

    fig, axis, _, _, _ = plotter.get_figure()
    vals.hist(bins=num_bins, ax=axis)
    plotter.save_figure(
        fig, filename=f"{param_name}-posterior", title_text=f"{param_name} posterior"
    )


def plot_loglikelihood_vs_parameter(
    plotter: Plotter, mcmc_tables: List[pd.DataFrame], param_name: str, burn_in=0
):
    """
    Plots the loglikelihood against parameter values.
    """
    ll_vals = mcmc_tables[0]["loglikelihood"]
    p_vals = mcmc_tables[0][param_name]
    log_ll_vals = [-log(-x) for x in ll_vals]
    fig, axis, _, _, _ = plotter.get_figure()
    axis.plot(p_vals[burn_in:], log_ll_vals[burn_in:], ".")
    axis.set_xlabel(param_name)
    axis.set_ylabel("-log(-loglikelihood)")
    plotter.save_figure(
        fig,
        filename=f"likelihood-against-{param_name}",
        title_text=f"likelihood against {param_name}",
    )


def sample_outputs_for_calibration_fit(
    output_name: str, mcmc_tables: List[pd.DataFrame], derived_output_tables: List[pd.DataFrame],
):
    assert len(mcmc_tables) == len(derived_output_tables)
    # For each chain grab 20 / top 100 accepted runs at random
    chosen_runs = []
    best_ll = -1.e16
    best_run = None
    best_chain_index = None
    for i, mcmc_table in enumerate(mcmc_tables):
        mask = mcmc_table["accept"] == 1
        num_accepted = len(mcmc_table[mask])
        choice_range = num_accepted // 3
        num_choices = min(choice_range, 20)
        runs = choices(mcmc_table[mask].idx.tolist()[-choice_range:], k=num_choices)
        chosen_runs.append(runs)

        this_chain_best_run, this_chain_best_ll = [
            np.argmax(mcmc_table.loglikelihood.tolist()),
            np.max(mcmc_table.loglikelihood.tolist())
        ]
        if this_chain_best_ll > best_ll:
            best_ll = this_chain_best_ll
            best_run = mcmc_table.idx.tolist()[this_chain_best_run]
            best_chain_index = i

    outputs = []
    for i, derived_output_table in enumerate(derived_output_tables):
        runs = chosen_runs[i]
        if i == best_chain_index:
            runs.append(best_run)

        for run in runs:
            mask = derived_output_table["idx"] == run
            times = derived_output_table[mask].times
            values = derived_output_table[mask][output_name]

            if run == best_run: # save it for the end of the outputs list
                mle_output = [copy.deepcopy(times), copy.deepcopy(values)]
            else:
                outputs.append([times, values])

    # automatically use the MLE run as the last chosen run
    outputs[-1] = mle_output
    return outputs, best_chain_index


def plot_calibration_fit(
    plotter: Plotter, output_name: str, outputs: list, best_chain_index, plot_config={}, is_logscale=False,
):
    fig, axis, _, _, _ = plotter.get_figure()

    for times, values in outputs:
        axis.plot(times, values)

    # re-plot MLE run with dotted line
    axis.plot(outputs[-1][0], outputs[-1][1], linestyle=(0, (1, 3)), color='black', linewidth=3)

    # Add plot targets
    output_config = {"name": output_name, "target_values": [], "target_times": []}
    outputs_to_plot = plot_config.get("outputs_to_plot", [])
    for o in outputs_to_plot:
        if o["name"] == output_name:
            output_config = o

    target_values = output_config["target_values"]
    target_times = output_config["target_times"]
    _plot_targets_to_axis(axis, target_values, target_times, on_uncertainty_plot=False)

    # Plot outputs
    axis.set_xlabel("time")
    axis.set_ylabel(output_name)
    if is_logscale:
        axis.set_yscale("log")

    if is_logscale:
        filename = f"calibration-fit-{output_name}-logscale"
        title_text = f"Calibration fit for {output_name} (logscale)"
    else:
        filename = f"calibration-fit-{output_name}"
        title_text = f"Calibration fit for {output_name}"

    # add text to indicate the best chain index
    axis.text(0.15, 0.9, 'MLE is from chain ' + str(best_chain_index),
              horizontalalignment='center', verticalalignment='center', transform=axis.transAxes)

    plotter.save_figure(fig, filename=filename, title_text=title_text)


def plot_timeseries_with_uncertainty_for_powerbi(
    plotter: Plotter,
    output_name: str,
    scenario_name: str,
    quantiles: dict,
    times: list,
    plot_config={},
):
    fig, axis, _, _, _ = plotter.get_figure()
    axis.fill_between(times, quantiles[0.025], quantiles[0.975], facecolor="lightsteelblue")
    axis.fill_between(times, quantiles[0.25], quantiles[0.75], facecolor="cornflowerblue")
    axis.plot(times, quantiles[0.50], color="navy")

    # Add plot targets
    output_config = {"name": output_name, "target_values": [], "target_times": []}
    outputs_to_plot = plot_config.get("outputs_to_plot", [])
    for o in outputs_to_plot:
        if o["name"] == output_name:
            output_config = o

    target_values = output_config["target_values"]
    target_times = output_config["target_times"]
    _plot_targets_to_axis(axis, target_values, target_times, on_uncertainty_plot=True)

    axis.set_xlabel("time")
    axis.set_ylabel(output_name)
    plotter.save_figure(
        fig,
        filename=f"uncertainty-{output_name}-{scenario_name}",
        title_text=f"{output_name} for {scenario_name}",
    )


def plot_timeseries_with_uncertainty(
    plotter: Plotter,
    path_to_percentile_outputs: str,
    output_name: str,
    scenario_indices,
    burn_in,
    plot_config={},
):
    percentile_db_path = os.path.join(
        path_to_percentile_outputs, "mcmc_percentiles_burned_" + str(burn_in) + ".db"
    )
    try:
        db = Database(percentile_db_path)
        output_perc = db.query(output_name)
    except:
        export_mcmc_quantiles(path_to_percentile_outputs, [output_name], burn_in=burn_in)
        db = Database(percentile_db_path)
        output_perc = db.query(output_name)

    fig, axis, _, _, _ = plotter.get_figure()
    scenario_list = "scenarios"
    for scenario_index in scenario_indices[::-1]:  # loop in reverse scenario order
        scenario_list += " " + str(scenario_index)
        df = output_perc[output_perc.Scenario == "S_" + str(scenario_index)]
        axis.fill_between(df.times, df.q_2_5, df.q_97_5, facecolor="lightsteelblue")
        axis.fill_between(df.times, df.q_25, df.q_75, facecolor="cornflowerblue")
        axis.plot(df.times, df.q_50, color="navy")
    try:
        output_config = next(o for o in plot_config["outputs_to_plot"] if o["name"] == output_name)
    except StopIteration:
        output_config = {"name": output_name, "target_values": [], "target_times": []}

    target_values = output_config["target_values"]
    # target_values = [[t_list[0] * 32.364904] for t_list in target_values] # for Malaysia ICU prev
    target_times = output_config["target_times"]
    _plot_targets_to_axis(axis, target_values, target_times, on_uncertainty_plot=True)
    axis.set_xlabel("time")
    axis.set_ylabel(output_name)

    plotter.save_figure(
        fig, filename=f"{output_name}-uncertainty-{scenario_list}", title_text=f"{output_name}",
    )


def _overwrite_non_accepted_mcmc_runs(mcmc_tables: List[pd.DataFrame], column_name: str):
    """
    Count non-accepted rows in a MCMC trace as the last accepted row.
    Modifies mcmc_tables in-place.
    """
    for table_df in mcmc_tables:
        prev_val = None
        for idx in range(len(table_df)):
            if table_df.at[idx, "accept"] == 1:
                prev_val = table_df.at[idx, column_name]
            else:
                table_df.at[idx, column_name] = prev_val


def plot_agg_compartments_multi_scenario(
    plotter: Plotter, scenarios: List[Scenario], compartment_names: List[str], is_logscale=False,
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

    plotter.save_figure(fig, filename="aggregate-compartments", title_text="aggregate compartments")


def plot_single_compartment_multi_scenario(
    plotter: Plotter, scenarios: List[Scenario], compartment_name: str, is_logscale=False,
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

    if len(legend) < 10:
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


def _plot_targets_to_axis(
    axis, target_values: List[float], target_times: List[int], on_uncertainty_plot=False
):
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
            axis.plot(x_vals, values[1:], "m", linewidth=1, color="red")

            axis.scatter(time, values[0], marker="o", color="red", s=30)
            axis.scatter(time, values[0], marker="o", color="white", s=10)
        else:
            # Plot a single point estimate
            value = values[0]
            if on_uncertainty_plot:
                axis.scatter(time, value, marker="o", color="black", s=10)
            else:
                axis.scatter(time, value, marker="o", color="red", s=30, zorder=999)
                axis.scatter(time, value, marker="o", color="white", s=10, zorder=999)


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
    filename = "exponential-growth-rate"
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


def plot_time_varying_input(
    plotter: Plotter,
    tv_key: str,
    tv_func: Callable[[float], float],
    times: List[float],
    is_logscale: bool,
):
    """
    Plot single simple plot of a function over time
    """
    # Plot requested func names.
    fig, axes, max_dims, n_rows, n_cols = plotter.get_figure()
    values = list(map(tv_func, times))
    if is_logscale:
        axes.set_yscale("log")

    axes.plot(times, values)
    plotter.save_figure(fig, filename=f"time-variant-{tv_key}", title_text=tv_key)


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
            get_country_mixing_matrix(location, "AUS"),
            yticklabels=model.mixing_categories,
            xticklabels=False,
            vmin=0.0,
            vmax=12.0,
        )
        plotter.save_figure(fig, f"mixing_matrix_{location}")


def plot_stacked_compartments_by_stratum(plotter: Plotter, scenarios: List[Scenario], compartment_name: str,
                                         stratify_by: str, multicountry=False, axis=None, config=2):
    models = [sc.model for sc in scenarios]
    times = (models[0].times[:-1] + models[1].times)

    phase_2_start = 214
    phase_2_end = {2: 398, 3: 580}

    if not multicountry:
        fig, axis, _, _, _ = plotter.get_figure()

    legend = []
    strata = models[0].all_stratifications[stratify_by]

    running_total = [0.] * len(times)

    blues = sns.color_palette("Blues_r", 4)
    reds = sns.color_palette("Oranges_r", 4)
    greens = sns.color_palette("BuGn_r", 4)
    purples = sns.cubehelix_palette(4)
    purples[0] = 'pink'

    # mark Phase 2 in the background:
    rect = patches.Rectangle((phase_2_start, 0), phase_2_end[config] - phase_2_start, 1.e9, linewidth=0,
                              facecolor='gold', alpha=.2)
    rect.set_zorder(1)

    # Add the patch to the Axes
    axis.add_patch(rect)

    strata_colors = blues + reds + greens + purples

    for color_idx, s in enumerate(strata):
        group_name = str(int(5.*color_idx))
        if color_idx < 15:
            group_name += "-" + str(int(5.*color_idx) + 4)
        else:
            group_name += "+"
        stratum_name = stratify_by + "_" + s

        if compartment_name in [c.split("X")[0] for c in models[0].compartment_names]:  # use outputs
            comp_names = [c for c in models[0].compartment_names if stratum_name in c.split('X') and compartment_name in c]
            comp_idx = [models[0].compartment_names.index(c) for c in comp_names]
            relevant_outputs_0 = models[0].outputs[:, comp_idx]
            values_0 = np.sum(relevant_outputs_0, axis=1)

            relevant_outputs_1 = models[1].outputs[:, comp_idx]
            values_1 = np.sum(relevant_outputs_1, axis=1)

            if compartment_name == 'recovered':
                deno_0 = np.sum(models[0].outputs, axis=1)
                values_0 = [100*v / d for (v, d) in zip(values_0, deno_0)]
                deno_1 = np.sum(models[1].outputs, axis=1)
                values_1 = [100*v / d for (v, d) in zip(values_1, deno_1)]

        else:  # use derived outputs
            relevant_output_names = [c for c in models[0].derived_outputs if stratum_name in c.split('X') and compartment_name in c]
            values_0 = [0] * len(models[0].times)
            values_1 = [0] * len(models[1].times)
            for out in relevant_output_names:
                values_0 = [v + d for (v, d) in zip(values_0, models[0].derived_outputs[out])]
                values_1 = [v + d for (v, d) in zip(values_1, models[1].derived_outputs[out])]

        new_running_total = [r + v for (r, v) in zip(running_total, list(values_0)[:-1] + list(values_1))]

        axis.fill_between(times, running_total, new_running_total, color=strata_colors[color_idx], label=group_name,
                          zorder=2, alpha=1.)
        legend.append(stratum_name)
        running_total = new_running_total

    max_val = max(running_total)
    axis.set_ylim((0, 1.1 * max_val))

    # axis.axvline(x=phase_2_start, linewidth=.8, dashes=[6, 4], color='black')
    # axis.axvline(x=phase_2_end[config],linewidth=.8, dashes=[6, 4], color='black')

    xticks = [61, 214, 398, 366 + 214]
    xlabs = ["1 Mar 2020", "1 Aug 2020", "1 Feb 2021", "1 Aug 2021"]

    axis.set_xlim((30, phase_2_end[config] + 90))
    axis.set_xticks(xticks)
    axis.set_xticklabels(xlabs, fontsize=12)

    for tick in axis.yaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    # axis.xaxis.get_major_ticks().label.set_fontsize(12)

    ylab = {
        "recovered": "% recovered",
        "incidence": "new diseased individuals",
        "infection_deaths": "number of deaths"
    }
    # axis.set_ylabel(ylab[compartment_name], fontsize=14)

    handles, labels = axis.get_legend_handles_labels()
    # axis.legend(reversed(handles), reversed(labels), bbox_to_anchor=(1.4, 1.1), title='Age:')

    if not multicountry:
        plotter.save_figure(fig, filename="compartments", title_text="compartments")

    return handles, labels


def plot_multicountry_rainbow(country_scenarios, config, mode, objective, include_config=False):
    fig_h = 20 if not include_config else 21
    heights = [1, 6, 6, 6, 6, 6, 6] if not include_config else [2, 1, 6, 6, 6, 6, 6, 6]

    fig = pyplot.figure(constrained_layout=True, figsize=(20, fig_h))  # (w, h)

    widths = [1, 6, 6, 6, 2]
    spec = fig.add_gridspec(ncols=5, nrows=len(heights), width_ratios=widths,
                            height_ratios=heights)

    output_names = ["incidence", "infection_deaths", "recovered"]
    output_titles = ["Daily disease incidence", "Daily deaths", "Percentage recovered"]

    countries = ['belgium', 'france', 'italy', 'spain', 'sweden', 'united-kingdom']
    country_names = [c.title() for c in countries]
    country_names[-1] = "United Kingdom"

    text_size = 23

    for i, country in enumerate(countries):
        i_grid = i + 2 if include_config else i+1
        for j, output in enumerate(output_names):
            ax = fig.add_subplot(spec[i_grid, j + 1])
            h, l = plot_stacked_compartments_by_stratum(None, country_scenarios[country], output, "agegroup",
                                                 multicountry=True, axis=ax, config=config)
            if i == 0:
                i_grid_output_labs = 1 if include_config else 0
                ax = fig.add_subplot(spec[i_grid_output_labs, j+1])
                ax.text(0.5, 0.5, output_titles[j], fontsize=text_size, horizontalalignment='center', verticalalignment='center')
                ax.axis("off")

        ax = fig.add_subplot(spec[i_grid, 0])
        ax.text(0.5, 0.5, country_names[i], rotation=90, fontsize=text_size, horizontalalignment='center', verticalalignment='center')
        ax.axis("off")

    if j == 2:
        ax = fig.add_subplot(spec[1:, 4])
        ax.legend(reversed(h), reversed(l), title='Age:', fontsize=15, title_fontsize=text_size,
                  labelspacing=1.0, loc='center')  # bbox_to_anchor=(1.4, 1.1),
        ax.axis("off")

    if include_config:
        ax = fig.add_subplot(spec[0, :])
        obj_name = 'deaths' if objective == 'deaths' else 'years of life lost'
        config_name = '6-month' if config == 2 else '12-month'

        config_label = "Optimisation by " + mode[3:] + " minimising " + obj_name + " with " + config_name + " mitigation"
        ax.text(0., 0.5, config_label, fontsize=text_size, horizontalalignment='left', verticalalignment='center')
        ax.axis("off")

    pyplot.rcParams["font.family"] = "Times New Roman"

    out_dir = "apps/covid_19/mixing_optimisation/opti_plots/figures/rainbows/"
    filename = out_dir + "rainbow_" + mode + "_config_" + str(config) + "_" + objective
    pyplot.savefig(filename + ".pdf")
    pyplot.savefig(filename + ".png", dpi=300)


def plot_hospital_occupancy(all_scenarios, country, mode, objective, ax, title):

    dash_style = {
        2: [6, 0],
        3: [6, 3]
    }

    colours = {
        "hospital_occupancy": sns.cubehelix_palette(4)[3],
        "icu_occupancy": sns.color_palette("Oranges_r", 4)[0]
    }

    x_min = 61. # 214


    # mark Phase 2 in the background:
    phase_2_start = 214
    phase_2_end = {2: 398, 3: 580}
    rect = patches.Rectangle((phase_2_start, 0), phase_2_end[config] - phase_2_start, 1.e9, linewidth=0,
                              facecolor='gold', alpha=.2)
    rect.set_zorder(1)

    for config in [2, 3]:
        scenarios = all_scenarios[mode][objective][config][country]
        models = [sc.model for sc in scenarios]
        times = (models[0].times + models[1].times)

        for output in ["hospital_occupancy", "icu_occupancy"]:
            if output == "hospital_occupancy":
                values_0 = models[0].derived_outputs[output]
                values_1 = models[1].derived_outputs[output]
            else:
                comp_names = [c for c in models[0].compartment_names if "clinical_icu" in c.split('X')]
                comp_idx = [models[0].compartment_names.index(c) for c in comp_names]
                relevant_outputs_0 = models[0].outputs[:, comp_idx]
                values_0 = np.sum(relevant_outputs_0, axis=1)

                relevant_outputs_1 = models[1].outputs[:, comp_idx]
                values_1 = np.sum(relevant_outputs_1, axis=1)

            values = list(values_0) + list(values_1)

            times = [t for t in times if t >= x_min]
            values = values[-len(times):]

            ax.plot(times, values, dashes=dash_style[config], color=colours[output], linewidth=2)

    ax.set_title(title)

    ax.set_ylabel("bed occupancy", fontsize=14)

    xticks = [214, 336, 366 + 91, 366 + 213]
    xlabs = ["1 Aug 2020", "1 Dec 2020", "1 Apr 2021", "1 Aug 2021"]

    ax.set_xlim((x_min, 366 + 213))
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabs, fontsize=12)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(12)


def plot_multicountry_hospital(all_scenarios, mode, objective):
    """
    Format of all_scenarios: all_scenarios[mode][objective][config][country]
    """
    fig = pyplot.figure(constrained_layout=True, figsize=(10, 9))  # (w, h)
    widths = [1, 1]
    heights = [1, 1, 1]
    spec = fig.add_gridspec(ncols=2, nrows=3, width_ratios=widths,
                            height_ratios=heights)

    countries = ['belgium', 'france', 'italy', 'spain', 'sweden', 'united-kingdom']
    country_names = [c.title() for c in countries]
    country_names[-1] = "United Kingdom"

    i_col = -1
    i_row = 0
    for i, country in enumerate(countries):
        i_col += 1
        if i_col >= 2:
            i_col = 0
            i_row += 1

        ax = fig.add_subplot(spec[i_row, i_col])
        plot_hospital_occupancy(all_scenarios, country, mode, objective, ax, country_names[i])

    pyplot.rcParams["font.family"] = "Times New Roman"

    out_dir = "apps/covid_19/mixing_optimisation/opti_plots/figures/hospitals/"
    filename = out_dir + "hospital_" + mode + "_" + objective
    # pyplot.savefig(filename + ".pdf")
    pyplot.savefig(filename + ".png", dpi=300)


def plot_optimal_plan(all_results, config, country, mode, ax):

    data = {
        'deaths': all_results['deaths'][config][country],
        'yoll': all_results['yoll'][config][country],
    }
    colours = {
        'deaths': 'lightcoral',
        'yoll': 'skyblue',
    }
    n_vars = {
        'by_age': 16,
        'by_location': 3,
    }
    bar_width = .32

    # arrows
    # arrow_par = {
    #     "by_age": {'width': .015, 'head_width': .06, 'head_length': .01},
    #     "by_location": {'width': .007, 'head_width': .03, 'head_length': .01},
    # }

    ax.grid(linewidth=.5, zorder=0, linestyle="dotted")
    ax.set_axisbelow(True)
    sensi_outputs = {}
    directions = ["down", "up"]
    for objective in ('deaths', 'yoll'):
        sensi_outputs[objective] = {}
        for direction in directions:
            path = "apps/covid_19/mixing_optimisation/optimisation_outputs/sensitivity/" +\
                   country + "_" + mode + "_" + str(config) + "_" + objective + "_" + direction + ".yml"
            try:
                with open(path, "r") as yaml_file:
                    sensi_outputs[objective][direction] = yaml.safe_load(yaml_file)
            except:
                sensi_outputs[objective][direction] = [0.] * n_vars[mode]

    ymax = 0.
    for i_age in range(n_vars[mode]):
        x_pos = i_age + 1.
        delta_xpos = -1
        for objective in ('deaths', 'yoll'):
            value = data[objective]['best_x' + str(i_age)].iloc[0]
            rect = patches.Rectangle((x_pos + delta_xpos * bar_width, 0.), bar_width, value, linewidth=.8,
                              facecolor=colours[objective], edgecolor='black', zorder=1)
            ax.add_patch(rect)

            # ax.plot((x_pos + delta_xpos * bar_width, x_pos + delta_xpos * bar_width + bar_width), (value, value),
            #         color=colours[objective])

            for direction in directions:
                arrow_length = sensi_outputs[objective][direction][i_age]
                if direction == "down":
                    arrow_length *= -1.
                # ax.arrow(x=x_pos + delta_xpos * bar_width + .5 * bar_width, y=value, dx=0, dy=arrow_length,
                #          color='black', length_includes_head=True,
                #          width=arrow_par[mode]['width'], head_width=arrow_par[mode]['head_width'],
                #          head_length=arrow_par[mode]['head_length'])
                _x = x_pos + delta_xpos * bar_width + .5 * bar_width
                ax.plot((_x, _x), (value, value + arrow_length), color='black', linewidth=2, zorder=3)

                ymax = max([ymax, value + arrow_length])

            delta_xpos = 0

    if mode == "by_age":
        # X axis settings
        major_ticks = [i + .5 for i in range(1, 16)]
        minor_ticks = range(1, 17)
        age_names = [ str(i*5) + "-" + str(i*5 + 4) for i in range(16)]
        age_names[-1] = "75+"

        ax.set_xticklabels(age_names, minor=True, rotation=45, fontsize=11)

        # Y axis settinds
        ylab = "Age-specific mixing factor"
    else:
        major_ticks = [1.5, 2.5]
        minor_ticks = [1, 2, 3]

        ylab = "Relative contact rate"
        ax.set_xticklabels(("other locations", "schools", "workplaces"), minor=True, fontsize=13)

    ax.axhline(y=1., color='black', dashes=(4, 6), linewidth=.8)

    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)

    ax.set_xticklabels("", major=True)
    ax.tick_params(axis="x", which="minor", length=0)
    ax.tick_params(axis="x", which="major", length=4)
    ax.set_xlim((0.5, n_vars[mode] + .5))

    ax.set_ylim((0, max((ymax, 1.05))))


    if config == 2:
        ax.set_ylabel(ylab, fontsize=14)


def plot_multicountry_optimal_plan(all_results, mode):
    pyplot.style.use("default")

    fig_width = {
        'by_age': 20,
        'by_location': 12
    }
    fig = pyplot.figure(constrained_layout=True, figsize=(fig_width[mode], 20))  # (w, h)
    pyplot.rcParams["font.family"] = "Times New Roman"

    widths = [1, 8, 8]
    heights = [1, 4, 4, 4, 4, 4, 4]
    spec = fig.add_gridspec(ncols=3, nrows=7, width_ratios=widths,
                            height_ratios=heights, hspace=0)
    text_size = 23

    countries = ['belgium', 'france', 'italy', 'spain', 'sweden', 'united-kingdom']
    country_names = [c.title() for c in countries]
    country_names[-1] = "United Kingdom"

    config_names = ("Six-month mitigation phase", "Twelve-month mitigation phase")

    for j, config in enumerate([2, 3]):
        for i, country in enumerate(countries):
            ax = fig.add_subplot(spec[i+1, j + 1])
            plot_optimal_plan(all_results, config, country, mode, ax)

            if j == 0:
                ax = fig.add_subplot(spec[i+1, 0])
                ax.text(0.8, 0.5, country_names[i], rotation=90, fontsize=text_size,
                        horizontalalignment='center', verticalalignment='center', fontweight='normal')
                ax.axis("off")

        ax = fig.add_subplot(spec[0, j + 1])
        ax.text(0.5, 0.5, config_names[j], fontsize=text_size, horizontalalignment='center', verticalalignment='center',
                fontweight='normal')
        ax.axis("off")

    out_dir = "apps/covid_19/mixing_optimisation/opti_plots/figures/optimal_plans/"
    filename = out_dir + "optimal_plan_" + mode
    # pyplot.savefig(filename + ".pdf")
    pyplot.savefig(filename + ".png", dpi=300)


def plot_mobility(all_values, country, loc_key, axis):

    times = list(range(215))
    values = all_values[country][loc_key]

    my_color = sns.cubehelix_palette(4)[-1]

    axis.plot(times, values, linewidth=2, color=my_color)

    xticks = [92, 153, 214]
    xlabs = ["1 Apr 2020", "1 Jun 2020", "1 Aug 2020"]

    axis.set_xlim((60, 214))
    axis.set_xticks(xticks)
    axis.set_xticklabels(xlabs, fontsize=12)

    axis.set_ylim((0., 1.1))


def plot_multicountry_mobility(all_values):
    fig = pyplot.figure(constrained_layout=True, figsize=(18, 20))  # (w, h)
    widths = [1, 6, 6, 6]
    heights = [1, 6, 6, 6, 6, 6, 6]
    spec = fig.add_gridspec(ncols=4, nrows=7, width_ratios=widths,
                            height_ratios=heights)

    location_names = {
        "other_locations": "other locations",
        "school": "schools",
        "work": "workplaces",
    }

    countries = ['belgium', 'france', 'italy', 'spain', 'sweden', 'united-kingdom']
    country_names = [c.title() for c in countries]
    country_names[-1] = "United Kingdom"

    text_size = 23

    for i, country in enumerate(countries):
        for j, loc_key in enumerate(["other_locations", "work", "school"]):
            ax = fig.add_subplot(spec[i+1, j + 1])
            plot_mobility(all_values, country, loc_key, axis=ax)
            if i == 0:
                ax = fig.add_subplot(spec[0, j+1])
                ax.text(0.5, 0.5, location_names[loc_key], fontsize=text_size, horizontalalignment='center', verticalalignment='center')
                ax.axis("off")

        ax = fig.add_subplot(spec[i+1, 0])
        ax.text(0.5, 0.5, country_names[i], rotation=90, fontsize=text_size, horizontalalignment='center', verticalalignment='center')
        ax.axis("off")

    pyplot.rcParams["font.family"] = "Times New Roman"

    out_dir = "apps/covid_19/mixing_optimisation/opti_plots/figures/inputs/"
    filename = out_dir + "mobility_inputs"
    # pyplot.savefig(filename + ".pdf")
    pyplot.savefig(filename + ".png", dpi=300)


def plot_hospital_uncertainty(pbi_outputs_dir, country, axis):
    dir_content = os.listdir(pbi_outputs_dir)
    for f in dir_content:
        if country in f:
            db_name = f

    db_path = os.path.join(pbi_outputs_dir, db_name)
    db = Database(db_path)
    uncertainty_df = db.query("uncertainty")
    quantile_vals = uncertainty_df["quantile"].unique().tolist()

    mask = uncertainty_df["type"] == 'hospital_occupancy'
    output_df = uncertainty_df[mask]
    data = {}
    max_plotted_time = 600.

    axis.grid(linewidth=1., zorder=0, linestyle="dotted")
    axis.set_axisbelow(True)

    max_q = 0.
    for scenario in range(5):
        sc_name = "S_" + str(scenario)
        data[sc_name] = {}
        mask = output_df["Scenario"] == sc_name
        scenario_df = output_df[mask]
        times = list(scenario_df.time.unique())[1:]
        if max_plotted_time in times:
            cut_tail = True
        else:
            cut_tail = False

        if cut_tail:
            max_time_index = times.index(max_plotted_time)
            times = times[:max_time_index + 1]

        quantiles = {}
        for q in quantile_vals:
            mask = scenario_df["quantile"] == q
            quantiles[q] = scenario_df[mask]["value"].tolist()[1:]
            if cut_tail:
                quantiles[q] = quantiles[q][:max_time_index + 1]
            max_q = max([max_q, max(quantiles[q])])
        data[sc_name]['quantiles'] = quantiles
        data[sc_name]['times'] = list(times)

    break_width = 13
    dates_to_tick = [
        [214., "1 Aug 20"],
        [366. + 32., "1 Feb 21"]
    ]
    x_ticks = [61]
    x_tick_labs = ['1 Mar 20']
    last_t_plotted = 0
    title_x = []
    for scenario in range(5):
        sc_name = "S_" + str(scenario)
        if scenario > 0:
            t_shift = last_t_plotted - data[sc_name]['times'][0]
            if scenario >= 2:
                t_shift += break_width
        else:
            t_shift = 0.

        times = [t_shift + t for t in data[sc_name]['times']]

        title_x.append(np.mean(times))

        for tick in dates_to_tick:
            requested_time = tick[0]
            requested_label = tick[1]
            graph_time = t_shift + requested_time
            x_ticks.append(graph_time)
            x_tick_labs.append(requested_label)

        if scenario > 0:
            axis.fill_between(times, data[sc_name]['quantiles'][0.025], data[sc_name]['quantiles'][0.975], facecolor="lightsteelblue")
            axis.fill_between(times, data[sc_name]['quantiles'][0.25], data[sc_name]['quantiles'][0.75], facecolor="cornflowerblue")
            axis.plot(times, data[sc_name]['quantiles'][0.50], color="navy")
        else:
            axis.fill_between(times, data[sc_name]['quantiles'][0.025], data[sc_name]['quantiles'][0.975], facecolor="plum")
            axis.fill_between(times, data[sc_name]['quantiles'][0.25], data[sc_name]['quantiles'][0.75], facecolor="mediumorchid")
            axis.plot(times, data[sc_name]['quantiles'][0.50], color="black")
        last_t_plotted = max(times)

        if scenario == 0:
            axis.axvline(x=last_t_plotted, color='black', dashes=(3, 3))
        else:
            rect = patches.Rectangle((last_t_plotted, 0), break_width, 1.e9, linewidth=0, hatch="/",
                              facecolor='peachpuff')
            rect.set_zorder(2)
            axis.add_patch(rect)

    x_lims = (50., last_t_plotted)
    axis.set_xlim(x_lims)
    axis.set_ylim((0, max_q * 1.05))

    axis.set_xticks(x_ticks)
    axis.set_xticklabels(x_tick_labs)
    axis.tick_params(axis="x", which="major", length=7)

    for tick in axis.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in axis.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)

    axis.set_ylabel("Hospital beds", fontsize=19)
    pyplot.margins(y=10.)

    return title_x, x_lims


def plot_multicountry_hospital_uncertainty(pbi_outputs_dir, immunity):
    fig = pyplot.figure(constrained_layout=True, figsize=(20, 20))  # (w, h)
    pyplot.style.use("default")
    pyplot.rcParams["hatch.linewidth"] = 1.


    widths = [1, 19]
    heights = [2, 6, 6, 6, 6, 6, 6]
    spec = fig.add_gridspec(ncols=2, nrows=7, width_ratios=widths,
                            height_ratios=heights)

    countries = ['belgium', 'france', 'italy', 'spain', 'sweden', 'united-kingdom']
    country_names = [c.title() for c in countries]
    country_names[-1] = "United Kingdom"

    titles_down = ["wave", "minimising deaths", "minimising YLLs", "minimising deaths", "minimising YLLs"]
    text_size = 23

    for i, country in enumerate(countries):
        ax = fig.add_subplot(spec[i+1, 0])
        ax.text(0.5, 0.5, country_names[i], rotation=90, fontsize=text_size, horizontalalignment='center', verticalalignment='center')
        ax.axis("off")

        ax = fig.add_subplot(spec[i+1, 1])
        title_x, x_lim = plot_hospital_uncertainty(pbi_outputs_dir, country, axis=ax)

        if i == 0:
            ax = fig.add_subplot(spec[0, 1])

            title_x[0] = 125.
            # Write "Past"
            ax.text(title_x[0], .8, "Past", fontsize=text_size, horizontalalignment='center', verticalalignment='center')
            # Write "6-month mitigation"
            ax.text(np.mean(title_x[1:3]), .8, "6-month mitigation", fontsize=text_size, horizontalalignment='center', verticalalignment='center')
            # Write "12-month mitigation"
            ax.text(np.mean(title_x[3:5]), .8, "12-month mitigation", fontsize=text_size, horizontalalignment='center', verticalalignment='center')

            for sc in range(5):
                ax.text(title_x[sc], 0.3, titles_down[sc], fontsize=text_size, horizontalalignment='center', verticalalignment='center')

            ax.axvline(x=213, color='black', ymin=.15, ymax=1.)
            ax.axvline(x=1002, color='black', ymin=.15, ymax=1.)


            ax.set_xlim(x_lim)
            ax.axis("off")

    pyplot.rcParams["font.family"] = "Times New Roman"

    out_dir = "apps/covid_19/mixing_optimisation/opti_plots/figures/hospital_uncertainty/"
    filename = out_dir + "hospital_uncertainty_by_age_" + immunity
    pyplot.savefig(filename + ".pdf")
    pyplot.savefig(filename + ".png", dpi=300)


def plot_mixing_matrix_opti(matrix, axis, fig, vmax, include_legend):

    axis.matshow(matrix)
    vmin = 0.
    # vmax_all = np.amax(self.model_runner.model_diagnostics[scenario]['contact_matrices'][key]['all'])
    # vmax = np.amax(self.model_runner.model_diagnostics[scenario]['contact_matrices'][key][location])

    pyplot.gca().xaxis.tick_bottom()

    with pyplot.style.context(('seaborn-dark')):
        im = axis.imshow(matrix.transpose(), cmap='hot', origin='lower', vmin=vmin, vmax=vmax)

        if include_legend:
            cbar = fig.colorbar(im, ax=axis)
            cbar.ax.set_ylabel('n contacts per day', rotation=90, va="bottom", fontsize=15)

            cbar.ax.yaxis.set_label_coords(4., 0.5)

        ticks = [i - .5 for i in range(16)]
        labs = [str(int(i*5.)) for i in range(16)]

        axis.set_xticks(ticks)
        axis.set_xticklabels(labs)
        axis.set_xlabel('index age')

        axis.set_yticks(ticks)
        axis.set_yticklabels(labs)
        axis.set_ylabel('contact age')


def plot_multicountry_optimised_matrices(original_matrices, optimised_matrices, mode, include_config=False):
    fig_h = 23 if not include_config else 25
    heights = [1, 6, 6, 6, 6, 6, 6] if not include_config else [2, 1, 6, 6, 6, 6, 6, 6]
    pyplot.rcParams["axes.grid"] = False

    fig = pyplot.figure(constrained_layout=True, figsize=(22, fig_h))  # (w, h)

    widths = [1, 5, 5, 5, 5, 5]
    spec = fig.add_gridspec(ncols=6, nrows=len(heights), width_ratios=widths,
                            height_ratios=heights, hspace=0, wspace=0)

    output_names = ['unmitigated', 'optimising deaths (6 mo.)', 'optimising deaths (12 mo.)', 'optimising YLLs (6 mo.)', 'optimising YLLs (6 mo.)']

    countries = ['belgium', 'france', 'italy', 'spain', 'sweden', 'united-kingdom']
    country_names = [c.title() for c in countries]
    country_names[-1] = "United Kingdom"

    text_size = 23

    for i, country in enumerate(countries):
        i_grid = i + 2 if include_config else i+1
        j = 0
        vmax = np.max(original_matrices[country])
        for objective in ["original", "deaths", "yoll"]:
            for config in [2, 3]:
                j += 1
                if objective == 'original':
                    matrix = original_matrices[country]
                else:
                    matrix = optimised_matrices[country][objective][config]
                ax = fig.add_subplot(spec[i_grid, j])
                include_legend = True if j == 5 else False
                plot_mixing_matrix_opti(matrix, ax, fig, vmax, include_legend)

                if i == 0:
                    i_grid_output_labs = 1 if include_config else 0
                    ax = fig.add_subplot(spec[i_grid_output_labs, j])
                    ax.text(0.5, 0.5, output_names[j - 1], fontsize=text_size, horizontalalignment='center', verticalalignment='center')
                    ax.axis("off")

                if objective == "original":
                    break

        ax = fig.add_subplot(spec[i_grid, 0])
        ax.text(0.5, 0.5, country_names[i], rotation=90, fontsize=text_size, horizontalalignment='center', verticalalignment='center')
        ax.axis("off")

    if include_config:
        ax = fig.add_subplot(spec[0, :])
        obj_name = 'deaths' if objective == 'deaths' else 'years of life lost'
        config_name = '6-month' if config == 2 else '12-month'

        config_label = "Optimisation by " + mode[3:]
        ax.text(0., 0.5, config_label, fontsize=text_size, horizontalalignment='left', verticalalignment='center')
        ax.axis("off")

    pyplot.rcParams["font.family"] = "Times New Roman"

    out_dir = "apps/covid_19/mixing_optimisation/opti_plots/figures/optimised_matrices/"
    filename = out_dir + "matrices_" + mode
    pyplot.savefig(filename + ".pdf")
    pyplot.savefig(filename + ".png", dpi=300)


def plot_waning_immunity_graph(scenarios: List[Scenario], output, axis, config):
    models = [sc.model for sc in scenarios]
    times = (models[0].times[:-1] + models[1].times)

    phase_2_start = 214
    phase_2_end = {2: 398, 3: 580}
    # mark Phase 2 in the background:
    rect = patches.Rectangle((phase_2_start, 0), phase_2_end[config] - phase_2_start, 1.e9, linewidth=0,
                              facecolor='gold', alpha=.2)
    rect.set_zorder(1)
    axis.add_patch(rect)

    scenario_names = [
        "baseline",
        "persistent immunity",
        "6 months immunity",
        "6 months immunity and\n50% less symptomatic",
        "24 months immunity",
        "24 months immunity and\n50% less symptomatic",
    ]

    sc_colors = ['black', 'black', 'mediumaquamarine', 'blue', 'lightcoral', 'crimson']

    y_max = 0.
    for scenario in [0, 2, 3, 4, 5, 1]:
        times = models[scenario].times
        values = models[scenario].derived_outputs[output]
        if scenario == 0:
            axis.plot(times, values, linewidth=3., color=sc_colors[scenario],
                      zorder=2)
        else:
            axis.plot(times, values, linewidth=3., label=scenario_names[scenario], color=sc_colors[scenario],
                      zorder=2)
        y_max = max([y_max, max(values)])
    xticks = [61, 214, 398, 366 + 214, 366 + 365]
    xlabs = ["1 Mar 20", "1 Aug 20", "1 Feb 21", "1 Aug 21", "31 Dec 21"]

    axis.set_xlim((30, max(times)))
    axis.set_xticks(xticks)
    axis.set_xticklabels(xlabs, fontsize=12)

    axis.set_ylim((0., y_max*1.05))


    for tick in axis.yaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    # axis.xaxis.get_major_ticks().label.set_fontsize(12)

    handles, labels = axis.get_legend_handles_labels()

    return handles, labels


def plot_multicountry_waning_immunity(country_scenarios, config, mode, objective, include_config=False):
    fig_h = 20 if not include_config else 21
    heights = [1, 6, 6, 6, 6, 6, 6] if not include_config else [2, 1, 6, 6, 6, 6, 6, 6]
    pyplot.rcParams["font.family"] = "Times New Roman"
    pyplot.style.use("default")
    fig = pyplot.figure(constrained_layout=True, figsize=(24, fig_h))  # (w, h)

    widths = [1, 6, 6, 6, 2]
    spec = fig.add_gridspec(ncols=5, nrows=len(heights), width_ratios=widths,
                            height_ratios=heights)

    output_names = ["incidence", "infection_deathsXall", "hospital_occupancy"]
    output_titles = ["Daily disease incidence", "Daily deaths", "Hospital occupancy"]

    countries = ['belgium', 'france', 'italy', 'spain', 'sweden', 'united-kingdom']
    country_names = [c.title() for c in countries]
    country_names[-1] = "United Kingdom"

    text_size = 23

    for i, country in enumerate(countries):
        i_grid = i + 2 if include_config else i+1
        for j, output in enumerate(output_names):
            ax = fig.add_subplot(spec[i_grid, j + 1])
            h, l = plot_waning_immunity_graph(country_scenarios[country], output, axis=ax, config=config)
            if i == 0:
                i_grid_output_labs = 1 if include_config else 0
                ax = fig.add_subplot(spec[i_grid_output_labs, j+1])
                ax.text(0.5, 0.5, output_titles[j], fontsize=text_size, horizontalalignment='center', verticalalignment='center')
                ax.axis("off")

        ax = fig.add_subplot(spec[i_grid, 0])
        ax.text(0.5, 0.5, country_names[i], rotation=90, fontsize=text_size, horizontalalignment='center', verticalalignment='center')
        ax.axis("off")

    if j == 2:
        ax = fig.add_subplot(spec[1:, 4])
        leg = ax.legend(h, l, fontsize=15, labelspacing=1.0, loc='center')  # bbox_to_anchor=(1.4, 1.1),
        for line in leg.get_lines():
            line.set_linewidth(5.0)
        ax.axis("off")

    if include_config:
        ax = fig.add_subplot(spec[0, :])
        obj_name = 'deaths' if objective == 'deaths' else 'years of life lost'
        config_name = '6-month' if config == 2 else '12-month'

        config_label = "Optimisation by " + mode[3:] + " minimising " + obj_name + " with " + config_name + " mitigation"
        ax.text(0., 0.5, config_label, fontsize=text_size, horizontalalignment='left', verticalalignment='center')
        ax.axis("off")


    out_dir = "apps/covid_19/mixing_optimisation/opti_plots/figures/waning_immunity/"
    filename = out_dir + "waning_immunity_" + mode + "_config_" + str(config) + "_" + objective
    pyplot.savefig(filename + ".pdf")
    pyplot.savefig(filename + ".png", dpi=300)




