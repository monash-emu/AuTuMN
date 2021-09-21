import os

from matplotlib import pyplot

from autumn.projects.tuberculosis.marshall_islands.outputs.utils import (
    OUTPUT_TITLES,
    get_format,
    make_output_directories,
    save_figure,
)
from autumn.tools.db.load import load_uncertainty_table
from autumn.tools.plots.uncertainty.plots import _plot_uncertainty
from autumn.tools.plots.utils import COLORS, _apply_transparency


def main(data_path, output_path):
    figure_path = os.path.join(output_path, "diabetes")
    make_output_directories(figure_path)
    get_format()
    uncertainty_df = load_uncertainty_table(data_path)
    plot_diabetes_graph(uncertainty_df, figure_path)


def plot_diabetes_graph(uncertainty_df, figure_path):

    outputs = ["incidence"]
    panel_h = 5
    panel_w = 7

    widths = [panel_w]
    heights = [panel_h] * len(outputs)
    fig = pyplot.figure(constrained_layout=True, figsize=(sum(widths), sum(heights)))  # (w, h)

    spec = fig.add_gridspec(ncols=1, nrows=len(outputs), width_ratios=widths, height_ratios=heights)

    diabetes_colours = {0: COLORS[6], 9: COLORS[2], 10: COLORS[7]}
    alphas = {0: 1, 9: 0.6, 10: 0.4}
    start_quantiles = {0: 2, 9: 1, 10: 1}
    linestyles = {0: "dashed", 9: "solid", 10: "solid"}

    for i, output in enumerate(outputs):
        ax = fig.add_subplot(spec[i, 0])

        for k, sc_id in enumerate([0, 9, 10]):
            sc_colors = _apply_transparency([diabetes_colours[sc_id]], [alphas[sc_id]])[0]

            _plot_uncertainty(
                ax,
                uncertainty_df,
                output,
                sc_id,
                2050,
                2020,
                sc_colors,
                start_quantile=start_quantiles[sc_id],
                zorder=k + 1,
                linestyle=linestyles[sc_id],
            )

        ax.set_ylabel(OUTPUT_TITLES[output], fontsize=17)

        ax.tick_params(axis="x", labelsize=15)
        ax.tick_params(axis="y", labelsize=15)
        ax.set_ylim(ymin=0)
        # plot_timeseries_with_uncertainty(
        #     None,
        #     uncertainty_df,
        #     output,
        #     scenario_idxs=[0, 9, 10],
        #     targets={},
        #     x_low=2020,
        #     x_up=2050,
        #     axis=ax,
        #     n_xticks=None,
        #     label_font_size=18,
        #     requested_x_ticks=None,
        #     show_title=False,
        #     ylab=ylab,
        #     x_axis_to_date=False,
        #     start_quantile=1
        # )

    save_figure("diabetes", figure_path)
