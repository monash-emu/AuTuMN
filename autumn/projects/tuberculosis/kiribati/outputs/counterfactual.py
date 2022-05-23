import os

from matplotlib import pyplot

from autumn.projects.tuberculosis.marshall_islands.outputs.utils import (
    OUTPUT_TITLES,
    REGION_TITLES,
    get_format,
    make_output_directories,
    save_figure,
)
from autumn.tools.db.load import load_uncertainty_table
from autumn.tools.plots.uncertainty.plots import plot_timeseries_with_uncertainty
from autumn.settings import BASE_PATH


def main(data_path, output_path):
    figure_path = os.path.join(output_path, "counterfactual")
    make_output_directories(figure_path)
    get_format()
    uncertainty_df = load_uncertainty_table(data_path)
    plot_counterfactual(uncertainty_df, figure_path)


def plot_counterfactual(uncertainty_df, figure_path):

    regions = ["majuro", "ebeye", "all"]
    outputs = ["incidence", "mortality", "percentage_latent", "notifications"]
    panel_h = 5
    panel_w = 7

    y_max = {
        "incidence": 900, "mortality": 250, "percentage_latent": 55, "notifications": 600
    }

    widths = [panel_w] * 3
    heights = [0.5] + [panel_h] * 4
    fig = pyplot.figure(constrained_layout=True, figsize=(sum(widths), sum(heights)))  # (w, h)
    spec = fig.add_gridspec(ncols=3, nrows=5, width_ratios=widths, height_ratios=heights)

    for j, region in enumerate(regions):
        ax = fig.add_subplot(spec[0, j])
        ax.text(
            0.5,
            0.5,
            REGION_TITLES[region],
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=20,
        )
        ax.set_xlim((0, 1))
        ax.set_ylim((0, 1))
        ax.axis("off")
        for i, output in enumerate(outputs):
            ax = fig.add_subplot(spec[i + 1, j])
            ylab = OUTPUT_TITLES[output] if j == 0 else None

            output_name = output if region == "all" else f"{output}Xlocation_{region}"

            plot_timeseries_with_uncertainty(
                None,
                uncertainty_df,
                output_name,
                scenario_idxs=[0, 1],
                targets={},
                x_low=2015,
                x_up=2050,
                axis=ax,
                n_xticks=None,
                label_font_size=18,
                requested_x_ticks=None,
                show_title=False,
                ylab=ylab,
                x_axis_to_date=False,
                start_quantile=0,
                overlay_uncertainty=True,
                legend=False
            )

            if output != "notifications":
                ax.set_ylim((0, y_max[output]))

    save_figure("counterfactual", figure_path)
