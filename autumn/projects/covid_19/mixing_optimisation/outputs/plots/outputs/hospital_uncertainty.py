import os

import matplotlib as mpl
import matplotlib.patches as patches
import numpy as np
from matplotlib import pyplot

from autumn.projects.covid_19.mixing_optimisation.constants import (
    DURATION_PHASES_2_AND_3,
    OPTI_REGIONS,
    PHASE_2_DURATION,
    PHASE_2_START_TIME,
)
from autumn.projects.covid_19.mixing_optimisation.mixing_opti import DURATIONS, MODES, OBJECTIVES
from autumn.projects.covid_19.mixing_optimisation.utils import (
    get_scenario_mapping,
    get_scenario_mapping_reverse,
)
from autumn.core.db.load import load_uncertainty_table
from autumn.settings import BASE_PATH

FIGURE_PATH = os.path.join(
    BASE_PATH,
    "apps",
    "covid_19",
    "mixing_optimisation",
    "outputs",
    "plots",
    "outputs",
    "figures",
    "hospital_uncertainty",
)

DATA_PATH = os.path.join(
    BASE_PATH,
    "apps",
    "covid_19",
    "mixing_optimisation",
    "outputs",
    "pbi_databases",
    "calibration_and_scenarios",
    "full_immunity",
)


def main():
    # Reset pyplot style
    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.pyplot.style.use("ggplot")

    uncertainty_dfs = {}
    for country in OPTI_REGIONS:
        dir_path = os.path.join(DATA_PATH, country)
        uncertainty_dfs[country] = load_uncertainty_table(dir_path)

    for mode in MODES:
        for output in ["hospital_occupancy", "proportion_seropositive"]:
            print(f"plotting {mode}, {output}")
            plot_multicountry_multiscenario_uncertainty(uncertainty_dfs, output, mode)


def plot_multiscenario_uncertainty(uncertainty_df, mode, axis, output, country):
    ylabs = {
        "infection_deathsXall": "daily number of deaths (weekly average)",
        "proportion_seropositive": "proportion recovered",
        "hospital_occupancy": "number of beds",
        "hospital_admissions": "new hospitalisations",
        "icu_occupancy": "ICU beds",
        "icu_admissions": "ICU admissions",
    }

    quantile_vals = uncertainty_df["quantile"].unique().tolist()
    mask = uncertainty_df["type"] == output
    output_df = uncertainty_df[mask]

    max_plotted_time = 600.0

    axis.grid(linewidth=1.0, zorder=0, linestyle="dotted")
    axis.set_axisbelow(True)

    max_q = 0.0

    data = {}
    sc_idx_to_plot = [0]

    for duration in DURATIONS:
        data[duration] = {}
        for objective in OBJECTIVES:
            sc_idx_to_plot.append(get_scenario_mapping_reverse(mode, duration, objective))

    for sc_idx in sc_idx_to_plot:
        data[sc_idx] = {}
        mask = output_df["scenario"] == sc_idx
        scenario_df = output_df[mask]
        times = list(scenario_df.time.unique())[1:]

        max_sc_time = PHASE_2_START_TIME if sc_idx == 0 else max_plotted_time

        if max_sc_time in times:
            cut_tail = True
        else:
            cut_tail = False

        if cut_tail:
            max_time_index = times.index(max_sc_time)
            times = times[: max_time_index + 1]

        quantiles = {}
        for q in quantile_vals:
            mask = scenario_df["quantile"] == q
            quantiles[q] = scenario_df[mask]["value"].tolist()[1:]
            if cut_tail:
                quantiles[q] = quantiles[q][: max_time_index + 1]
            max_q = max([max_q, max(quantiles[q])])
        data[sc_idx]["quantiles"] = quantiles
        data[sc_idx]["times"] = list(times)

    break_width = 13
    dates_to_tick = [[275.0, "1 Oct 20"], [366.0 + 91.0, "1 Apr 21"]]
    x_ticks = [61]
    x_tick_labs = ["1 Mar 20"]
    last_t_plotted = 0
    title_x = []
    for i, sc_idx in enumerate(sc_idx_to_plot):
        if sc_idx > 0:
            t_shift = last_t_plotted - data[sc_idx]["times"][0]
            if i >= 2:
                t_shift += break_width
        else:
            t_shift = 0.0

        times = [t_shift + t for t in data[sc_idx]["times"]]

        title_x.append(np.mean(times))

        for tick in dates_to_tick:
            requested_time = tick[0]
            requested_label = tick[1]
            graph_time = t_shift + requested_time
            x_ticks.append(graph_time)
            x_tick_labs.append(requested_label)

        if sc_idx > 0:
            axis.fill_between(
                times,
                data[sc_idx]["quantiles"][0.025],
                data[sc_idx]["quantiles"][0.975],
                facecolor="lightsteelblue",
            )
            axis.fill_between(
                times,
                data[sc_idx]["quantiles"][0.25],
                data[sc_idx]["quantiles"][0.75],
                facecolor="cornflowerblue",
            )
            axis.plot(times, data[sc_idx]["quantiles"][0.50], color="navy")
        else:
            axis.fill_between(
                times,
                data[sc_idx]["quantiles"][0.025],
                data[sc_idx]["quantiles"][0.975],
                facecolor="plum",
            )
            axis.fill_between(
                times,
                data[sc_idx]["quantiles"][0.25],
                data[sc_idx]["quantiles"][0.75],
                facecolor="mediumorchid",
            )
            axis.plot(times, data[sc_idx]["quantiles"][0.50], color="black")
        last_t_plotted = max(times)

        if sc_idx == 0:
            axis.axvline(x=last_t_plotted, color="black", dashes=(3, 3))
        else:
            rect = patches.Rectangle(
                (last_t_plotted, 0),
                break_width,
                1.0e9,
                linewidth=0,
                hatch="/",
                facecolor="peachpuff",
            )
            rect.set_zorder(2)
            axis.add_patch(rect)

    x_lims = (50.0, last_t_plotted)
    axis.set_xlim(x_lims)
    axis.set_ylim((0, max_q * 1.05))

    axis.set_xticks(x_ticks)
    axis.set_xticklabels(x_tick_labs)
    axis.tick_params(axis="x", which="major", length=7)

    for tick in axis.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in axis.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)

    axis.set_ylabel(ylabs[output], fontsize=19)
    pyplot.margins(y=10.0)

    return title_x, x_lims


def plot_multicountry_multiscenario_uncertainty(uncertainty_dfs, output, mode):
    fig = pyplot.figure(constrained_layout=True, figsize=(20, 20))  # (w, h)
    pyplot.style.use("default")
    pyplot.rcParams["hatch.linewidth"] = 1.0

    widths = [1, 19]
    heights = [2, 6, 6, 6, 6, 6, 6]
    spec = fig.add_gridspec(ncols=2, nrows=7, width_ratios=widths, height_ratios=heights)

    countries = ["belgium", "france", "italy", "spain", "sweden", "united-kingdom"]
    country_names = [c.title() for c in countries]
    country_names[-1] = "United Kingdom"

    titles_down = [
        "epidemic",
        "minimising deaths",
        "minimising YLLs",
        "minimising deaths",
        "minimising YLLs",
    ]
    text_size = 23

    for i, country in enumerate(countries):
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

        ax = fig.add_subplot(spec[i + 1, 1])
        title_x, x_lim = plot_multiscenario_uncertainty(
            uncertainty_dfs[country], mode, ax, output, country
        )

        if i == 0:
            ax = fig.add_subplot(spec[0, 1])

            title_x[0] = 125.0
            # Write "Past"
            ax.text(
                title_x[0],
                0.8,
                "Past",
                fontsize=text_size,
                horizontalalignment="center",
                verticalalignment="center",
            )
            # Write "6-month mitigation"
            ax.text(
                np.mean(title_x[1:3]),
                0.8,
                "6-month mitigation",
                fontsize=text_size,
                horizontalalignment="center",
                verticalalignment="center",
            )
            # Write "12-month mitigation"
            ax.text(
                np.mean(title_x[3:5]),
                0.8,
                "12-month mitigation",
                fontsize=text_size,
                horizontalalignment="center",
                verticalalignment="center",
            )

            for sc in range(len(title_x)):
                ax.text(
                    title_x[sc],
                    0.3,
                    titles_down[sc],
                    fontsize=text_size,
                    horizontalalignment="center",
                    verticalalignment="center",
                )

            ax.axvline(x=275, color="black", ymin=0.15, ymax=1.0)
            ax.axvline(x=950, color="black", ymin=0.15, ymax=1.0)

            ax.set_xlim(x_lim)
            ax.axis("off")

    pyplot.rcParams["font.family"] = "Times New Roman"

    filename = f"{output}_uncertainty_{mode}"
    png_path = os.path.join(FIGURE_PATH, f"{filename}.png")
    pdf_path = os.path.join(FIGURE_PATH, f"{filename}.pdf")
    pyplot.savefig(png_path, dpi=300)
    pyplot.savefig(pdf_path)


if __name__ == "__main__":
    main()
