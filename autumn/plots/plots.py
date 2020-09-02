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
from autumn.db.database import Database
import matplotlib.gridspec as gridspec


from .plotter import Plotter, COLOR_THEME

logger = logging.getLogger(__name__)


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
    best_ll = -1.0e16
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
            np.max(mcmc_table.loglikelihood.tolist()),
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

            if run == best_run:  # save the MLE run for the end of the outputs list
                mle_output = [copy.deepcopy(times), copy.deepcopy(values)]
            else:
                outputs.append([times, values])

    # automatically use the MLE run as the last chosen run
    outputs[-1] = mle_output
    return outputs, best_chain_index


def plot_calibration_fit(
    plotter: Plotter, output_name: str, outputs: list, best_chain_index, targets, is_logscale=False,
):
    fig, axis, _, _, _ = plotter.get_figure()

    # Track the maximum value being plotted
    max_value = 0.0

    for times, values in outputs:
        axis.plot(times, values)
        max_value = max(values) if max(values) > max_value else max_value

    # Mark the MLE run with a dotted line
    axis.plot(outputs[-1][0], outputs[-1][1], linestyle=(0, (1, 3)), color="black", linewidth=3)

    # Add plot targets
    output_config = {"output_key": output_name, "values": [], "times": []}
    for t in targets.values():
        if t["output_key"] == output_name:
            output_config = t

    values = output_config["values"]
    times = output_config["times"]
    _plot_targets_to_axis(axis, values, times, on_uncertainty_plot=False)

    # Find upper limit for y-axis
    if values:
        upper_buffer = 2.0
        max_target = max(values)
        upper_ylim = (
            max_value if max_value < max_target * upper_buffer else max_target * upper_buffer
        )
    else:
        upper_ylim = max_value

    # Plot outputs
    axis.set_xlabel("time")
    axis.set_ylabel(output_name)
    if is_logscale:
        axis.set_yscale("log")
    else:
        axis.set_ylim([0.0, upper_ylim])

    if is_logscale:
        filename = f"calibration-fit-{output_name}-logscale"
        title_text = f"Calibration fit for {output_name} (logscale)"
    else:
        filename = f"calibration-fit-{output_name}"
        title_text = f"Calibration fit for {output_name}"

    # add text to indicate the best chain index
    axis.text(
        0.15,
        0.9,
        "MLE is from chain " + str(best_chain_index),
        horizontalalignment="center",
        verticalalignment="center",
        transform=axis.transAxes,
    )

    plotter.save_figure(fig, filename=filename, title_text=title_text)


def plot_timeseries_with_uncertainty(
    plotter: Plotter, output_name: str, scenario_name: str, quantiles: dict, times: list, targets,
):
    fig, axis, _, _, _ = plotter.get_figure()
    axis.fill_between(times, quantiles[0.025], quantiles[0.975], facecolor="lightsteelblue")
    axis.fill_between(times, quantiles[0.25], quantiles[0.75], facecolor="cornflowerblue")
    axis.plot(times, quantiles[0.50], color="navy")

    # Add plot targets
    output_config = {"values": [], "times": []}
    for t in targets.values():
        if t["output_key"] == output_name:
            output_config = t

    values = output_config["values"]
    times = output_config["times"]
    _plot_targets_to_axis(axis, values, times, on_uncertainty_plot=True)

    axis.set_xlabel("time")
    axis.set_ylabel(output_name)
    plotter.save_figure(
        fig,
        filename=f"uncertainty-{output_name}-{scenario_name}",
        title_text=f"{output_name} for {scenario_name}",
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
    output_name = output_config["output_key"]
    legend = []
    for idx, scenario in enumerate(reversed(scenarios)):
        color_idx = len(scenarios) - idx - 1
        _plot_outputs_to_axis(axis, scenario, output_name, color_idx=color_idx, alpha=0.7)
        legend.append(scenario.name)

    axis.legend(legend)
    values = output_config["values"]
    times = output_config["times"]
    _plot_targets_to_axis(axis, values, times)
    if is_logscale:
        axis.set_yscale("log")

    plotter.save_figure(fig, filename=output_name, title_text=output_name)


def plot_outputs_single(
    plotter: Plotter, scenario: Scenario, output_config: dict, is_logscale=False
):
    """
    Plot the model derived/generated outputs requested by the user for a single scenario.
    """
    fig, axis, _, _, _ = plotter.get_figure()
    if is_logscale:
        axis.set_yscale("log")

    output_name = output_config["output_key"]
    target_values = output_config["values"]
    target_times = output_config["times"]
    _plot_outputs_to_axis(axis, scenario, output_name)
    _plot_targets_to_axis(axis, target_values, target_times)
    plotter.save_figure(fig, filename=output_name, subdir="outputs", title_text=output_name)


def _plot_outputs_to_axis(axis, scenario: Scenario, name: str, color_idx=0, alpha=1):
    """
    Plot outputs requested by output_config from scenario to the provided axis.
    """
    model = scenario.model
    plot_values = model.derived_outputs[name]
    # Plot the values as a line.
    axis.plot(model.times, plot_values, color=COLOR_THEME[color_idx], alpha=alpha)


def _plot_targets_to_axis(axis, values: List[float], times: List[int], on_uncertainty_plot=False):
    """
    Plot output value calibration targets as points on the axis.
    # TODO: add back ability to plot confidence interval
    x_vals = [time, time]
    axis.plot(x_vals, values[1:], "m", linewidth=1, color="red")
    axis.scatter(time, values[0], marker="o", color="red", s=30)
    axis.scatter(time, values[0], marker="o", color="white", s=10)
    """
    assert len(times) == len(values), "Targets have inconsistent length"
    # Plot a single point estimate
    if on_uncertainty_plot:
        axis.scatter(times, values, marker="o", color="black", s=10)
    else:
        axis.scatter(times, values, marker="o", color="red", s=30, zorder=999)
        axis.scatter(times, values, marker="o", color="white", s=10, zorder=999)


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
    if is_logscale:
        axes.set_yscale("log")

    if type(tv_func) is not list:
        funcs = [tv_func]
    else:
        funcs = tv_func

    for func in funcs:
        values = list(map(func, times))
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

    if model._static_mixing_matrix is None:
        logger.debug("No static mixing matrix found, skipping model.")
        return

    fig, axis, max_dims, n_rows, n_cols = plotter.get_figure()

    axis = sns.heatmap(
        model._static_mixing_matrix,
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


def plot_stacked_compartments_by_stratum(
    plotter: Plotter,
    scenarios: List[Scenario],
    compartment_name: str,
    stratify_by: str,
    multicountry=False,
    axis=None,
    config=2,
):
    models = [sc.model for sc in scenarios]
    times = models[0].times + models[1].times

    if not multicountry:
        fig, axis, _, _, _ = plotter.get_figure()

    legend = []
    strata = models[0].all_stratifications[stratify_by]

    running_total = [0.0] * len(times)

    blues = sns.color_palette("Blues_r", 4)
    reds = sns.color_palette("Oranges_r", 4)
    greens = sns.color_palette("BuGn_r", 4)
    purples = sns.cubehelix_palette(4)
    purples[0] = "pink"

    strata_colors = blues + reds + greens + purples

    for color_idx, s in enumerate(strata):
        group_name = str(int(5.0 * color_idx))
        if color_idx < 15:
            group_name += "-" + str(int(5.0 * color_idx) + 4)
        else:
            group_name += "+"
        stratum_name = stratify_by + "_" + s

        if compartment_name in [
            c.split("X")[0] for c in models[0].compartment_names
        ]:  # use outputs
            comp_names = [
                c
                for c in models[0].compartment_names
                if stratum_name in c.split("X") and compartment_name in c
            ]
            comp_idx = [models[0].compartment_names.index(c) for c in comp_names]
            relevant_outputs_0 = models[0].outputs[:, comp_idx]
            values_0 = np.sum(relevant_outputs_0, axis=1)

            relevant_outputs_1 = models[1].outputs[:, comp_idx]
            values_1 = np.sum(relevant_outputs_1, axis=1)

            if compartment_name == "recovered":
                deno_0 = np.sum(models[0].outputs, axis=1)
                values_0 = [100 * v / d for (v, d) in zip(values_0, deno_0)]
                deno_1 = np.sum(models[1].outputs, axis=1)
                values_1 = [100 * v / d for (v, d) in zip(values_1, deno_1)]

        else:  # use derived outputs
            relevant_output_names = [
                c
                for c in models[0].derived_outputs
                if stratum_name in c.split("X") and compartment_name in c
            ]
            values_0 = [0] * len(models[0].times)
            values_1 = [0] * len(models[1].times)
            for out in relevant_output_names:
                values_0 = [v + d for (v, d) in zip(values_0, models[0].derived_outputs[out])]
                values_1 = [v + d for (v, d) in zip(values_1, models[1].derived_outputs[out])]

        new_running_total = [
            r + v for (r, v) in zip(running_total, list(values_0) + list(values_1))
        ]

        axis.fill_between(
            times,
            running_total,
            new_running_total,
            color=strata_colors[color_idx],
            label=group_name,
        )
        legend.append(stratum_name)
        running_total = new_running_total

    phase_2_end = {2: 398, 3: 580}

    axis.axvline(x=214, linewidth=0.8, dashes=[6, 4], color="black")
    axis.axvline(x=phase_2_end[config], linewidth=0.8, dashes=[6, 4], color="black")

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
        "infection_deaths": "number of deaths",
    }
    # axis.set_ylabel(ylab[compartment_name], fontsize=14)

    handles, labels = axis.get_legend_handles_labels()
    # axis.legend(reversed(handles), reversed(labels), bbox_to_anchor=(1.4, 1.1), title='Age:')

    if not multicountry:
        plotter.save_figure(fig, filename="compartments", title_text="compartments")

    return handles, labels


def plot_multicountry_rainbow(country_scenarios, config, mode, objective):
    fig = pyplot.figure(constrained_layout=True, figsize=(20, 20))  # (w, h)
    widths = [1, 6, 6, 6, 2]
    heights = [1, 6, 6, 6, 6, 6, 6]
    spec = fig.add_gridspec(ncols=5, nrows=7, width_ratios=widths, height_ratios=heights)

    output_names = ["incidence", "infection_deaths", "recovered"]
    output_titles = ["Daily disease incidence", "Daily deaths", "Percentage recovered"]

    countries = ["belgium", "france", "italy", "spain", "sweden", "united-kingdom"]
    country_names = [c.title() for c in countries]
    country_names[-1] = "United Kingdom"

    text_size = 23

    for i, country in enumerate(countries):
        for j, output in enumerate(output_names):
            ax = fig.add_subplot(spec[i + 1, j + 1])
            h, l = plot_stacked_compartments_by_stratum(
                None,
                country_scenarios[country],
                output,
                "agegroup",
                multicountry=True,
                axis=ax,
                config=config,
            )
            if i == 0:
                ax = fig.add_subplot(spec[0, j + 1])
                ax.text(
                    0.5,
                    0.5,
                    output_titles[j],
                    fontsize=text_size,
                    horizontalalignment="center",
                    verticalalignment="center",
                )
                ax.axis("off")

        ax = fig.add_subplot(spec[i + 1, 0])
        ax.text(
            0.5,
            0.5,
            country_names[i],
            rotation=90,
            fontsize=text_size,
            horizontalalignment="center",
            verticalalignment="center",
        )
        ax.axis("off")

    if j == 2:
        ax = fig.add_subplot(spec[1:, 4])
        ax.legend(
            reversed(h),
            reversed(l),
            title="Age:",
            fontsize=15,
            title_fontsize=text_size,
            labelspacing=1.0,
            loc="center",
        )  # bbox_to_anchor=(1.4, 1.1),
        ax.axis("off")

    out_dir = "apps/covid_19/mixing_optimisation/opti_plots/figures/rainbows/"
    filename = out_dir + "rainbow_" + mode + "_config_" + str(config) + "_" + objective
    pyplot.savefig(filename + ".pdf")
    pyplot.savefig(filename + ".png", dpi=300)


def plot_hospital_occupancy(all_scenarios, country, mode, objective, ax, title):

    dash_style = {2: [6, 0], 3: [6, 3]}

    colours = {
        "hospital_occupancy": sns.cubehelix_palette(4)[3],
        "icu_occupancy": sns.color_palette("Oranges_r", 4)[0],
    }

    x_min = 214

    for config in [2, 3]:
        scenarios = all_scenarios[mode][objective][config][country]
        models = [sc.model for sc in scenarios]
        times = models[0].times + models[1].times

        for output in ["hospital_occupancy", "icu_occupancy"]:
            if output == "hospital_occupancy":
                values_0 = models[0].derived_outputs[output]
                values_1 = models[1].derived_outputs[output]
            else:
                comp_names = [
                    c for c in models[0].compartment_names if "clinical_icu" in c.split("X")
                ]
                comp_idx = [models[0].compartment_names.index(c) for c in comp_names]
                relevant_outputs_0 = models[0].outputs[:, comp_idx]
                values_0 = np.sum(relevant_outputs_0, axis=1)

                relevant_outputs_1 = models[1].outputs[:, comp_idx]
                values_1 = np.sum(relevant_outputs_1, axis=1)

            values = list(values_0) + list(values_1)

            times = [t for t in times if t >= x_min]
            values = values[-len(times) :]

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
    spec = fig.add_gridspec(ncols=2, nrows=3, width_ratios=widths, height_ratios=heights)

    countries = ["belgium", "france", "italy", "spain", "sweden", "united-kingdom"]
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

    out_dir = "apps/covid_19/mixing_optimisation/opti_plots/figures/hospitals/"
    filename = out_dir + "rainbow_" + mode + "_" + objective
    pyplot.savefig(filename + ".pdf")
    pyplot.savefig(filename + ".png", dpi=300)
