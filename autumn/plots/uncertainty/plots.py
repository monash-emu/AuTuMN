"""
Plotting projection uncertainty.
"""
from typing import List
import logging
import datetime

import pandas as pd
from math import ceil

from autumn.plots.plotter import Plotter
from autumn.plots.calibration.plots import _plot_targets_to_axis
from matplotlib import colors, pyplot
from autumn.plots.utils import change_xaxis_to_date

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
        n_xticks=None,
        ref_date=datetime.date(2019, 12, 31),
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

    # Sort out x-axis
    change_xaxis_to_date(axis, ref_date)
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


def plot_seroprevalence_by_age(
    plotter: Plotter,
    uncertainty_df: pd.DataFrame,
    scenario_id: int,
    time: float,
    ref_date=datetime.date(2019, 12, 31)
):
    fig, axis, _, _, _, _ = plotter.get_figure()
    mask = (
        (uncertainty_df["scenario"] == scenario_id)
        & (uncertainty_df["time"] == time)
    )
    df = uncertainty_df[mask]
    quantile_vals = df["quantile"].unique().tolist()
    seroprevalence_by_age = {}
    sero_outputs = [output for output in df["type"].unique().tolist() if "proportion_seropositiveXagegroup_" in output]
    if len(sero_outputs) == 0:
        return

    for output in sero_outputs:
        output_mask = df["type"] == output
        age = output.split("proportion_seropositiveXagegroup_")[1]
        seroprevalence_by_age[age] = {}
        for q in quantile_vals:
            q_mask = df["quantile"] == q
            seroprevalence_by_age[age][q] = [100. * v for v in df[output_mask][q_mask]["value"].tolist()]

    q_keys = sorted(quantile_vals)
    num_quantiles = len(q_keys)
    half_length = num_quantiles // 2

    for age in list(seroprevalence_by_age.keys()):
        x_pos = 2.5 + float(age)
        axis.plot([x_pos, x_pos], [seroprevalence_by_age[age][q_keys[0]], seroprevalence_by_age[age][q_keys[-1]]],
                  "-", color='black', lw=.5)

        if num_quantiles % 2:
            q_key = q_keys[half_length]
            axis.plot(x_pos, seroprevalence_by_age[age][q_key], 'o', color='black', markersize=2)

    axis.set_xlabel('age (years)', fontsize=10)
    axis.set_ylabel('% seropositive', fontsize=10)

    _date = ref_date + datetime.timedelta(days=time)

    axis.set_title(f'seroprevalence on {_date}', fontsize=12)
    plotter.save_figure(fig, filename='sero_by_age', subdir="outputs", title_text='')


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
