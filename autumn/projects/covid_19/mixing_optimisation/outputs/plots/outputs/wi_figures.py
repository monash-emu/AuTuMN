import os

import matplotlib as mpl
import matplotlib.patches as patches
import numpy as np
import seaborn as sns
import yaml
from matplotlib import pyplot

from autumn.projects.covid_19.mixing_optimisation.constants import (
    DURATION_PHASES_2_AND_3,
    OPTI_REGIONS,
    PHASE_2_DURATION,
    PHASE_2_START_TIME,
)
from autumn.projects.covid_19.mixing_optimisation.mixing_opti import DURATIONS, MODES, OBJECTIVES
from autumn.projects.covid_19.mixing_optimisation.outputs.plots.outputs.rainbows import (
    apply_scenario_mask,
    get_output_data,
)
from autumn.projects.covid_19.mixing_optimisation.utils import get_wi_scenario_mapping_reverse
from autumn.coredb import Database
from autumn.coredb.load import find_db_paths, load_derived_output_tables
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
    "wi_figures",
)

DATA_PATH = os.path.join(
    BASE_PATH, "apps", "covid_19", "mixing_optimisation", "outputs", "pbi_databases", "manual_runs"
)

IMMUNITY_MODES = ["full_immunity", "short_severe", "short_milder", "long_severe", "long_milder"]


def main():
    # Reset pyplot style
    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.pyplot.style.use("ggplot")

    all_derived_outputs = {}
    for immunity_mode in IMMUNITY_MODES:
        data_folder = os.path.join(DATA_PATH, immunity_mode)
        derived_outputs, _ = get_output_data(data_folder)
        all_derived_outputs[immunity_mode] = derived_outputs

    for pessimistic in [False, True]:
        for duration in DURATIONS:
            for objective in OBJECTIVES:
                print(f"Plot {pessimistic}, {duration}, {objective}")
                plot_multicountry_waning_immunity(
                    all_derived_outputs,
                    duration,
                    objective,
                    include_config=False,
                    pessimistic=pessimistic,
                )


def plot_waning_immunity_graph(
    all_derived_outputs, output, country, duration, objective, axis, pessimistic=False
):
    sc_info = {
        "all_wi_scenarios": {
            "sc_names": [
                "baseline",
                "persistent immunity",
                "6 months immunity",
                "6 months immunity and\n50% less symptomatic",
                "24 months immunity",
                "24 months immunity and\n50% less symptomatic",
            ],
            "sc_colors": ["black", "black", "mediumaquamarine", "blue", "lightcoral", "crimson"],
            "immunity_modes": [
                "full_immunity",
                "full_immunity",
                "short_severe",
                "short_milder",
                "long_severe",
                "long_milder",
            ],
            "sc_order": [0, 2, 3, 4, 5, 1],
        },
        "pessimistic": {
            "sc_names": [
                "baseline",
                "70% mixing factor",
                "80% mixing factor",
                "90% mixing factor",
                "no mixing reduction",
            ],
            "sc_colors": ["black", "black", "blue", "cornflowerblue", "deepskyblue"],
            "final_mixings": [1.0, 0.7, 0.8, 0.9, 1.0],
            "sc_order": [0, 4, 3, 2, 1],
        },
    }
    type_key = "pessimistic" if pessimistic else "all_wi_scenarios"

    phase_2_start = PHASE_2_START_TIME

    # mark Phase 2 in the background:
    rect = patches.Rectangle(
        (phase_2_start, -1000),
        PHASE_2_DURATION[duration],
        2.0e9,
        linewidth=0,
        facecolor="gold",
        alpha=0.2,
    )
    rect.set_zorder(1)
    axis.add_patch(rect)
    axis.axhline(y=0, linewidth=0.5, color="grey")

    y_max = 0.0
    for scenario in sc_info[type_key]["sc_order"][1:]:
        if pessimistic:
            final_mixing = sc_info[type_key]["final_mixings"][scenario]
            immunity_mode = "short_severe"
        else:
            final_mixing = 1.0
            immunity_mode = sc_info[type_key]["immunity_modes"][scenario]

        derived_output_df = all_derived_outputs[immunity_mode][country]

        if scenario == 0:
            run_sc_idx = 0
        else:
            run_sc_idx = get_wi_scenario_mapping_reverse(duration, objective, final_mixing)

        sc_mask = derived_output_df["scenario"] == run_sc_idx
        df = derived_output_df[sc_mask]

        times = list(df.times)
        values = list(df[output])

        if scenario == 0:
            axis.plot(
                times,
                values,
                linewidth=3.0,
                color=sc_info[type_key]["sc_colors"][scenario],
                zorder=2,
            )
        else:
            axis.plot(
                times,
                values,
                linewidth=3.0,
                label=sc_info[type_key]["sc_names"][scenario],
                color=sc_info[type_key]["sc_colors"][scenario],
                zorder=2,
            )
        y_max = max([y_max, max(values)])

    phase_2_end_date = {DURATIONS[0]: "1 Apr 21", DURATIONS[1]: "1 Oct 21"}

    xticks = [PHASE_2_START_TIME, PHASE_2_START_TIME + PHASE_2_DURATION[duration], 366 + 365]
    xlabs = ["1 Oct 20", phase_2_end_date[duration], "31 Dec 21"]

    axis.set_xlim((PHASE_2_START_TIME, max(times)))
    axis.set_xticks(xticks)
    axis.set_xticklabels(xlabs, fontsize=12)

    axis.set_ylim((-0.05 * y_max, y_max * 1.05))

    for tick in axis.yaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    # axis.xaxis.get_major_ticks().label.set_fontsize(12)

    handles, labels = axis.get_legend_handles_labels()

    return handles, labels


def plot_multicountry_waning_immunity(
    all_derived_outputs, duration, objective, include_config=False, pessimistic=False
):
    fig_h = 20 if not include_config else 21
    heights = [1, 6, 6, 6, 6, 6, 6] if not include_config else [2, 1, 6, 6, 6, 6, 6, 6]
    pyplot.rcParams["font.family"] = "Times New Roman"
    pyplot.style.use("default")
    fig = pyplot.figure(constrained_layout=True, figsize=(24, fig_h))  # (w, h)

    widths = [1, 6, 6, 6, 2]
    spec = fig.add_gridspec(ncols=5, nrows=len(heights), width_ratios=widths, height_ratios=heights)

    output_names = ["incidence", "infection_deaths", "hospital_occupancy"]
    output_titles = ["Daily disease incidence", "Daily deaths", "Hospital occupancy"]

    countries = ["belgium", "france", "italy", "spain", "sweden", "united-kingdom"]
    country_names = [c.title() for c in countries]
    country_names[-1] = "United Kingdom"

    text_size = 23

    for i, country in enumerate(countries):
        i_grid = i + 2 if include_config else i + 1
        for j, output in enumerate(output_names):
            ax = fig.add_subplot(spec[i_grid, j + 1])
            h, l = plot_waning_immunity_graph(
                all_derived_outputs,
                output,
                country,
                duration=duration,
                objective=objective,
                axis=ax,
                pessimistic=pessimistic,
            )
            if i == 0:
                i_grid_output_labs = 1 if include_config else 0
                ax = fig.add_subplot(spec[i_grid_output_labs, j + 1])
                ax.text(
                    0.5,
                    0.5,
                    output_titles[j],
                    fontsize=text_size,
                    horizontalalignment="center",
                    verticalalignment="center",
                )
                ax.axis("off")

        ax = fig.add_subplot(spec[i_grid, 0])
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

    if j == 2:
        ax = fig.add_subplot(spec[1:, 4])
        leg = ax.legend(
            h, l, fontsize=15, labelspacing=1.0, loc="center"
        )  # bbox_to_anchor=(1.4, 1.1),
        for line in leg.get_lines():
            line.set_linewidth(5.0)
        ax.axis("off")

    if include_config:
        ax = fig.add_subplot(spec[0, :])
        obj_name = "deaths" if objective == "deaths" else "years of life lost"
        config_name = "6-month" if duration == DURATIONS[0] else "12-month"

        config_label = (
            "Optimisation by by age minimising " + obj_name + " with " + config_name + " mitigation"
        )
        ax.text(
            0.0,
            0.5,
            config_label,
            fontsize=text_size,
            horizontalalignment="left",
            verticalalignment="center",
        )
        ax.axis("off")

    filename = f"waning_immunity_by_age_{duration}_{objective}"
    folder = "pessimistic" if pessimistic else "all_wi_scenarios"
    path = os.path.join(FIGURE_PATH, folder)

    png_path = os.path.join(path, f"{filename}.png")
    pdf_path = os.path.join(path, f"{filename}.pdf")
    pyplot.savefig(png_path, dpi=300)
    pyplot.savefig(pdf_path)


if __name__ == "__main__":
    main()
