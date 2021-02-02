from matplotlib import pyplot
import os


from settings import BASE_PATH
from autumn.db.load import load_uncertainty_table
from autumn.plots.uncertainty.plots import _plot_uncertainty
from autumn.plots.utils import COLORS, _apply_transparency

from apps.tuberculosis.regions.marshall_islands.outputs.utils import (
    OUTPUT_TITLES,
    INTERVENTION_TITLES,
    save_figure,
    get_format,
    make_output_directories,
)


FIGURE_PATH = os.path.join(
    BASE_PATH,
    "apps",
    "tuberculosis",
    "regions",
    "marshall_islands",
    "outputs",
    "figures",
    "elimination",
)

DATA_PATH = os.path.join(
    BASE_PATH, "apps", "tuberculosis", "regions", "marshall_islands", "outputs", "pbi_databases"
)

end_tb_targets = {
    "mortality": {
        2025: 43.25,  # 75% reduction compared to 2015 (173)
        2035: 8.65,  # 95% reduction compared to 2015 (173)
    },
    "incidence": {
        2025: 253.5,  # 50% reduction compared to 2015 (507)
        2035: 50.7,  # 90% reduction compared to 2015 (507)
    },
}
target_colours = {2025: "limegreen", 2035: "green"}


def main():
    make_output_directories(FIGURE_PATH)
    get_format()
    uncertainty_df = load_uncertainty_table(DATA_PATH)
    for is_logscale in [True, False]:
        plot_elimination(uncertainty_df, is_logscale)


def plot_elimination(uncertainty_df, is_logscale=False):

    interventions = ["ACF", "ACF_LTBI", "hh_pt"]
    scenario_idxs = {"ACF": [0, 4, 3], "ACF_LTBI": [0, 6, 5], "hh_pt": [0, 9]}
    colors_idx = {"ACF": [0, 7, 1], "ACF_LTBI": [0, 7, 1], "hh_pt": [0, 4]}
    alphas = {"ACF": [1.0, 0.8, 0.7], "ACF_LTBI": [1.0, 0.8, 0.7], "hh_pt": [1.0, 0.7]}

    if is_logscale:
        outputs = ["incidence", "mortality"]
    else:
        outputs = ["incidence", "mortality", "notifications"]
    panel_h = 5
    panel_w = 7

    widths = [panel_w] * len(interventions)
    heights = [0.5] + [panel_h] * len(outputs)
    fig = pyplot.figure(constrained_layout=True, figsize=(sum(widths), sum(heights)))  # (w, h)
    spec = fig.add_gridspec(
        ncols=len(interventions), nrows=len(outputs) + 1, width_ratios=widths, height_ratios=heights
    )

    for j, intervention in enumerate(interventions):
        ax = fig.add_subplot(spec[0, j])
        ax.text(
            0.5,
            0.5,
            INTERVENTION_TITLES[intervention],
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=20,
        )
        ax.set_xlim((0, 1))
        ax.set_ylim((0, 1))
        ax.axis("off")
        for i, output in enumerate(outputs):
            ax = fig.add_subplot(spec[i + 1, j])
            sc_colors = [COLORS[h] for h in colors_idx[intervention]]
            sc_colors = _apply_transparency(sc_colors, alphas[intervention])

            for k, scenario_idx in enumerate(scenario_idxs[intervention]):
                x_low = 2015 if scenario_idx == 0 else 2020
                _plot_uncertainty(
                    ax,
                    uncertainty_df,
                    output,
                    scenario_idx,
                    2050,
                    x_low,
                    sc_colors[k],
                    start_quantile=1,
                    zorder=k + 1,
                )
            if j == 0:
                ax.set_ylabel(OUTPUT_TITLES[output], fontsize=20)

            ax.tick_params(axis="x", labelsize=18)
            ax.tick_params(axis="y", labelsize=18)

            if output in ["incidence", "mortality"]:
                for year, value in end_tb_targets[output].items():
                    ax.plot(float(year), value, marker="o", color=target_colours[year])

            if is_logscale:
                ax.set_yscale("log")
                ax.set_ylim((10 ** -3, 10 ** 3))
                if output == "incidence":
                    ax.hlines(y=1, xmin=2015, xmax=2050, colors="black", linestyle="dashed")
                    ax.text(2016, 0.6, "pre-elimination threshold", fontsize=12)
            else:
                ax.set_ylim(ymin=0)
    filename = "elimination"
    if is_logscale:
        filename += "_logscale"

    save_figure(filename, FIGURE_PATH)


if __name__ == "__main__":
    main()
