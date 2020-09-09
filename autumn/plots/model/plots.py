"""
Plots for a model that has been run.
"""
import os
import logging
from typing import List, Callable

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot
import matplotlib.gridspec as gridspec

from autumn.tool_kit.scenarios import Scenario


from autumn.plots.plotter import Plotter, COLOR_THEME

logger = logging.getLogger(__name__)


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
        _plot_hospital_occupancy(all_scenarios, country, mode, objective, ax, country_names[i])

    out_dir = "apps/covid_19/mixing_optimisation/opti_plots/figures/hospitals/"
    filename = out_dir + "rainbow_" + mode + "_" + objective
    pyplot.savefig(filename + ".pdf")
    pyplot.savefig(filename + ".png", dpi=300)


def _plot_hospital_occupancy(all_scenarios, country, mode, objective, ax, title):

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

