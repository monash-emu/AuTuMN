from matplotlib import pyplot
import os


from autumn.constants import BASE_PATH
from autumn.db.load import load_uncertainty_table
from autumn.plots.uncertainty.plots import plot_timeseries_with_uncertainty

from apps.tuberculosis.regions.marshall_islands.outputs.utils import OUTPUT_TITLES, REGION_TITLES, save_figure, get_format, make_output_directories

FIGURE_PATH = os.path.join(
    BASE_PATH, "apps", "tuberculosis", "regions", "marshall_islands", "outputs", "figures", "counterfactual"
)

DATA_PATH = os.path.join(
    BASE_PATH, "apps", "tuberculosis", "regions", "marshall_islands", "outputs", "pbi_databases"
)


def main():
    make_output_directories(FIGURE_PATH)

    get_format()
    uncertainty_df = load_uncertainty_table(DATA_PATH)
    plot_counterfactual(uncertainty_df)


def plot_counterfactual(uncertainty_df):

    regions = ["majuro", "ebeye", "all"]
    outputs = ["incidence", "mortality", "percentage_latent", "notifications"]
    panel_h = 5
    panel_w = 7

    widths = [panel_w] * 3
    heights = [0.5] + [panel_h] * 4
    fig = pyplot.figure(constrained_layout=True, figsize=(sum(widths), sum(heights)))  # (w, h)
    spec = fig.add_gridspec(ncols=3, nrows=5, width_ratios=widths,
                            height_ratios=heights)

    for j, region in enumerate(regions):
        ax = fig.add_subplot(spec[0, j])
        ax.text(
            0.5, 0.5, REGION_TITLES[region],  horizontalalignment='center', verticalalignment='center', fontsize=20
        )
        ax.set_xlim((0, 1))
        ax.set_ylim((0, 1))
        ax.axis("off")
        for i, output in enumerate(outputs):
            ax = fig.add_subplot(spec[i+1, j])
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
                start_quantile=1
            )

    save_figure("counterfactual", FIGURE_PATH)


if __name__ == "__main__":
    main()

