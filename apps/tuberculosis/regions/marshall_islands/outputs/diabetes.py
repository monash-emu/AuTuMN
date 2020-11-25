from matplotlib import pyplot
import os


from autumn.constants import BASE_PATH
from autumn.db.load import load_uncertainty_table
from autumn.plots.uncertainty.plots import plot_timeseries_with_uncertainty, _plot_uncertainty
from autumn.plots.utils import COLORS, ALPHAS, _apply_transparency

from apps.tuberculosis.regions.marshall_islands.outputs.utils import OUTPUT_TITLES, REGION_TITLES, save_figure, get_format

FIGURE_PATH = os.path.join(
    BASE_PATH, "apps", "tuberculosis", "regions", "marshall_islands", "outputs", "figures", "diabetes"
)

DATA_PATH = os.path.join(
    BASE_PATH, "apps", "tuberculosis", "regions", "marshall_islands", "outputs", "pbi_databases"
)


def main():
    get_format()
    uncertainty_df = load_uncertainty_table(DATA_PATH)
    plot_diabetes_graph(uncertainty_df)


def plot_diabetes_graph(uncertainty_df):

    outputs = ["incidence"]
    panel_h = 5
    panel_w = 7

    widths = [panel_w]
    heights = [panel_h] * len(outputs)
    fig = pyplot.figure(constrained_layout=True, figsize=(sum(widths), sum(heights)))  # (w, h)
    spec = fig.add_gridspec(ncols=1, nrows=len(outputs), width_ratios=widths,
                            height_ratios=heights)

    diabetes_colours = {
        0: COLORS[6],
        7: COLORS[2],
        8: COLORS[7]
    }
    alphas = {
        0: 1,
        7: .6,
        8: .4
    }
    start_quantiles = {
        0: 2,
        7: 1,
        8: 1
    }
    linestyles = {
        0: 'dashed',
        7: 'solid',
        8: 'solid'
    }

    for i, output in enumerate(outputs):
        ax = fig.add_subplot(spec[i, 0])

        for k, sc_id in enumerate([0, 7, 8]):
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
                        zorder=k+1,
                        linestyle=linestyles[sc_id]
                    )

        ax.set_ylabel(OUTPUT_TITLES[output], fontsize=17)

        ax.tick_params(axis="x", labelsize=15)
        ax.tick_params(axis="y", labelsize=15)
        ax.set_ylim(ymin=0)
        # plot_timeseries_with_uncertainty(
        #     None,
        #     uncertainty_df,
        #     output,
        #     scenario_idxs=[0, 7, 8],
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

    save_figure("diabetes", FIGURE_PATH)


if __name__ == "__main__":
    main()

