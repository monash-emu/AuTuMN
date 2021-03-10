from matplotlib import pyplot
import os


from settings import BASE_PATH
from autumn.db.load import load_uncertainty_table
from autumn.plots.uncertainty.plots import (
    _plot_uncertainty,
    _get_target_values,
    _plot_targets_to_axis,
)
from autumn.plots.utils import COLORS
from autumn.tool_kit.params import load_targets

from apps.tuberculosis.regions.marshall_islands.outputs.utils import (
    OUTPUT_TITLES,
    save_figure,
    get_format,
    make_output_directories,
)
from apps.tuberculosis.regions.marshall_islands.calibrate import targets_to_use

FIGURE_PATH = os.path.join(
    BASE_PATH,
    "apps",
    "tuberculosis",
    "regions",
    "marshall_islands",
    "outputs",
    "figures",
    "calibration",
)

DATA_PATH = os.path.join(
    BASE_PATH, "apps", "tuberculosis", "regions", "marshall_islands", "outputs", "pbi_databases"
)


def main():
    make_output_directories(FIGURE_PATH)
    get_format()
    uncertainty_df = load_uncertainty_table(DATA_PATH)
    plot_screening_rate(uncertainty_df)
    plot_model_fits(uncertainty_df)


def plot_screening_rate(uncertainty_df):
    n_col = 1
    n_row = 1

    outputs = ["screening_rate"]
    panel_h = 5
    panel_w = 7

    widths = [panel_w] * n_col
    heights = [panel_h] * n_row
    fig = pyplot.figure(constrained_layout=True, figsize=(sum(widths), sum(heights)))  # (w, h)
    spec = fig.add_gridspec(ncols=n_col, nrows=n_row, width_ratios=widths, height_ratios=heights)

    # load targets
    targets = load_targets("tuberculosis", "marshall_islands")

    x_low = 1950
    x_up = 2050

    i_row = 0
    i_col = 0

    plotted_scenario_ranges = {
        0: [x_low, x_up],
    }

    for output in outputs:
        ax = fig.add_subplot(spec[i_row, i_col])

        for sc_idx, sc_range in plotted_scenario_ranges.items():
            _plot_uncertainty(
                ax,
                uncertainty_df,
                output,
                sc_idx,
                sc_range[1],
                sc_range[0],
                COLORS[0],
                start_quantile=0,
            )

        ax.set_ylabel(OUTPUT_TITLES[output], fontsize=20)

        ax.tick_params(axis="x", labelsize=18)
        ax.tick_params(axis="y", labelsize=18)
        ax.set_ylim(ymin=0)

        i_col += 1
        if i_col == n_col:
            i_col = 0
            i_row += 1

        values, times = _get_target_values(targets, output)
        trunc_values = [v for (v, t) in zip(values, times) if x_low <= t <= x_up]
        trunc_times = [t for (v, t) in zip(values, times) if x_low <= t <= x_up]
        _plot_targets_to_axis(ax, trunc_values, trunc_times, on_uncertainty_plot=True)

    save_figure("screening_rate", FIGURE_PATH)


def plot_model_fits(uncertainty_df):
    n_col = 3
    n_row = 2

    outputs = targets_to_use
    panel_h = 5
    panel_w = 7

    widths = [panel_w] * n_col
    heights = [panel_h] * n_row
    fig = pyplot.figure(constrained_layout=True, figsize=(sum(widths), sum(heights)))  # (w, h)
    spec = fig.add_gridspec(ncols=n_col, nrows=n_row, width_ratios=widths, height_ratios=heights)

    # load targets
    targets = load_targets("tuberculosis", "marshall_islands")

    x_low = 2010
    x_up = 2020

    i_row = 0
    i_col = 0

    plotted_scenario_ranges = {
        0: [x_low, x_up],
        #        1: [2016, x_up],
    }

    for output in outputs:
        ax = fig.add_subplot(spec[i_row, i_col])

        for sc_idx, sc_range in plotted_scenario_ranges.items():
            _plot_uncertainty(
                ax,
                uncertainty_df,
                output,
                sc_idx,
                sc_range[1],
                2009,  # sc_range[0],
                COLORS[0],
                start_quantile=0,
            )

        ax.set_ylabel(OUTPUT_TITLES[output], fontsize=20)

        ax.tick_params(axis="x", labelsize=18)
        ax.tick_params(axis="y", labelsize=18)
        ax.set_ylim(ymin=0)

        i_col += 1
        if i_col == n_col:
            i_col = 0
            i_row += 1

        values, times = _get_target_values(targets, output)
        trunc_values = [v for (v, t) in zip(values, times) if x_low <= t <= x_up]
        trunc_times = [t for (v, t) in zip(values, times) if x_low <= t <= x_up]
        _plot_targets_to_axis(ax, trunc_values, trunc_times, on_uncertainty_plot=True)

    save_figure("model_fits", FIGURE_PATH)


if __name__ == "__main__":
    main()
