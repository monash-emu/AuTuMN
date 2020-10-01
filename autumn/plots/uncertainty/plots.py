"""
Plotting projection uncertainty.
"""
import logging
from datetime import datetime

import pandas as pd

from autumn.plots.plotter import Plotter
from autumn.plots.calibration.plots import _plot_targets_to_axis
from matplotlib import colors

logger = logging.getLogger(__name__)


def make_colors_transparent(raw_color_list, alphas):
    for i in range(len(raw_color_list)):
        for j in range(len(raw_color_list[i])):
            rgb_color = list(colors.colorConverter.to_rgb(raw_color_list[i][j]))
            raw_color_list[i][j] = rgb_color + [alphas[i]]
    return raw_color_list


def plot_timeseries_with_uncertainty(
    plotter: Plotter,
    uncertainty_df: pd.DataFrame,
    output_name: str,
    scenarios: list,
    targets: dict,
    is_logscale=False,
    x_low=0.,
    x_up=2000.
):
    fig, axis, _, _, _, _ = plotter.get_figure()
    title = plotter.get_plot_title(output_name)
    # Plot quantiles
    colors = (
        ["lightsteelblue", "cornflowerblue", "royalblue", "navy"],
        ["plum", "mediumorchid", "darkviolet", "black"],
    )
    alphas = (1., .6)
    colors = make_colors_transparent(colors, alphas)

    for scenario in scenarios:
        color_index = min(scenario, 1)
        mask = (uncertainty_df["type"] == output_name) & (uncertainty_df["scenario"] == scenario) &\
               (uncertainty_df["time"] <= x_up) & (uncertainty_df["time"] >= x_low)
        df = uncertainty_df[mask]
        times = df.time.unique()
        quantiles = {}
        quantile_vals = df["quantile"].unique().tolist()
        for q in quantile_vals:
            mask = df["quantile"] == q
            quantiles[q] = df[mask]["value"].tolist()
        q_keys = sorted([float(k) for k in quantiles.keys()])
        num_quantiles = len(q_keys)
        half_length = num_quantiles // 2
        for i in range(half_length):
            color = colors[color_index][i]
            start_key = q_keys[i]
            end_key = q_keys[-(i + 1)]
            axis.fill_between(times, quantiles[start_key], quantiles[end_key], facecolor=color)

        if num_quantiles % 2:
            q_key = q_keys[half_length]
            axis.plot(times, quantiles[q_key], color=colors[color_index][3])

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
    if is_logscale:
        axis.set_yscale("log")
    else:
        axis.set_ylim(ymin=0)

    scenarios_string = ", ".join([str(t) for t in scenarios])
    scenario_title = "baseline scenario" if scenarios == [0] else "Scenarios " + scenarios_string
    plotter.save_figure(
        fig,
        filename=f"uncertainty-{output_name}",
        title_text=f"{title} for {scenario_title}",
    )
