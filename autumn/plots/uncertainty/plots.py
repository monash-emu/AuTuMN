"""
Plotting projection uncertainty.
"""
from typing import List
import logging
from datetime import datetime

import pandas as pd
from math import ceil

from autumn.plots.plotter import Plotter
from autumn.plots.calibration.plots import _plot_targets_to_axis
from matplotlib import colors, pyplot
from matplotlib.ticker import FormatStrFormatter

logger = logging.getLogger(__name__)

ALPHAS = (1.0, 0.6)
COLORS = (
    ["lightsteelblue", "cornflowerblue", "royalblue", "navy"],
    ["plum", "mediumorchid", "darkviolet", "black"],
)


def plot_timeseries_with_uncertainty(
    plotter: Plotter,
    uncertainty_df: pd.DataFrame,
    output_name: str,
    scenario_idxs: List[int],
    targets: dict,
    is_logscale=False,
    x_low=0.0,
    x_up=1e6,
    axis=None,
    n_xticks=None
):
    """
    Plots the uncertainty timeseries for one or more scenarios.
    Also plots any calibration targets that are provided.
    """
    single_panel = axis is None
    if single_panel:
        fig, axis, _, _, _, _ = plotter.get_figure()
    colors = _apply_transparency(COLORS, ALPHAS)

    # Plot each scenario on a single axis.
    for scenario_idx in scenario_idxs:
        color_idx = min(scenario_idx, 1)
        scenario_colors = colors[color_idx]
        _plot_uncertainty(
            axis, uncertainty_df, output_name, scenario_idx, x_up, x_low, scenario_colors
        )

    # Add plot targets
    values, times = _get_target_values(targets, output_name)
    _plot_targets_to_axis(axis, values, times, on_uncertainty_plot=True)

    axis.set_xlabel("time")
    if n_xticks is not None:
        pyplot.locator_params(axis='x', nbins=n_xticks)

    output_title = plotter.get_plot_title(output_name)
    axis.set_ylabel(output_title)
    if is_logscale:
        axis.set_yscale("log")
    elif not (output_name.startswith('rel_diff') or output_name.startswith('abs_diff')):
        axis.set_ylim(ymin=0)

    if scenario_idxs == [0]:
        title = f"{output_title} for baseline scenario"
    else:
        scenarios_string = ", ".join(map(str, scenario_idxs))
        title = f"{output_title} for scenarios {scenarios_string}"

    if single_panel:
        idx_str = "-".join(map(str, scenario_idxs))
        filename = f"uncertainty-{output_name}-{idx_str}"
        plotter.save_figure(fig, filename=filename, title_text=title)


def _plot_uncertainty(
    axis,
    uncertainty_df: pd.DataFrame,
    output_name: str,
    scenario_idx: int,
    x_up: float,
    x_low: float,
    colors: List[str],
):
    """Plots the uncertainty values in the provided dataframe to an axis"""
    mask = (
        (uncertainty_df["type"] == output_name)
        & (uncertainty_df["scenario"] == scenario_idx)
        & (uncertainty_df["time"] <= x_up)
        & (uncertainty_df["time"] >= x_low)
    )
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
        color = colors[i]
        start_key = q_keys[i]
        end_key = q_keys[-(i + 1)]
        axis.fill_between(times, quantiles[start_key], quantiles[end_key], facecolor=color)

    if num_quantiles % 2:
        q_key = q_keys[half_length]
        axis.plot(times, quantiles[q_key], color=colors[3])


def plot_multi_output_timeseries_with_uncertainty(
    plotter: Plotter,
    uncertainty_df: pd.DataFrame,
    output_names: str,
    scenarios: list,
    all_targets: dict,
    is_logscale=False,
    x_low=0.,
    x_up=2000.,
    n_xticks=None
):
    if len(output_names) * len(scenarios) == 0:
        return
    # pyplot.rcParams.update({'font.size': 15})

    max_n_col = 2
    n_panels = len(output_names)
    n_cols = min(max_n_col, n_panels)
    n_rows = ceil(n_panels / max_n_col)

    fig = pyplot.figure(constrained_layout=True, figsize=(n_cols * 7, n_rows * 5))  # (w, h)
    spec = fig.add_gridspec(ncols=n_cols, nrows=n_rows)

    i_col = 0
    i_row = 0
    for output_name in output_names:
        targets = {k: v for k, v in all_targets.items() if v["output_key"] == output_name}
        ax = fig.add_subplot(spec[i_row, i_col])
        plot_timeseries_with_uncertainty(
            plotter, uncertainty_df, output_name, scenarios, targets, is_logscale, x_low, x_up, ax, n_xticks
        )
        i_col += 1
        if i_col == max_n_col:
            i_col = 0
            i_row += 1

    plotter.save_figure(fig, filename='multi_uncertainty', subdir="outputs", title_text='')

    # out_dir = "apps/tuberculosis/regions/marshall_islands/figures/calibration_targets/"
    # filename = out_dir + "targets"
    # pyplot.savefig(filename + ".pdf")


def _get_target_values(targets: dict, output_name: str):
    """Pulls out values for a given target"""
    output_config = {"values": [], "times": []}
    for t in targets.values():
        if t["output_key"] == output_name:
            output_config = t

    values = output_config["values"]
    times = output_config["times"]
    return values, times


def _apply_transparency(color_list: List[str], alphas: List[str]):
    """Make a list of colours transparent, based on a list of alphas"""
    for i in range(len(color_list)):
        for j in range(len(color_list[i])):
            rgb_color = list(colors.colorConverter.to_rgb(color_list[i][j]))
            color_list[i][j] = rgb_color + [alphas[i]]
    return color_list
