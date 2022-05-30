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
from autumn.projects.covid_19.mixing_optimisation.utils import get_scenario_mapping_reverse
from autumn.coret Database
from autumn.core.load import find_db_paths, load_derived_output_tables
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
    "rainbows",
)

DATA_PATH = os.path.join(
    BASE_PATH,
    "apps",
    "covid_19",
    "mixing_optimisation",
    "outputs",
    "pbi_databases",
    "manual_runs_full_immunity",
)


def main():
    # Reset pyplot style
    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.pyplot.style.use("ggplot")
    all_derived_outputs, all_outputs = get_output_data()

    for mode in MODES:
        for duration in DURATIONS:
            for objective in OBJECTIVES:
                plot_multicountry_rainbow(
                    all_derived_outputs,
                    all_outputs,
                    mode,
                    duration,
                    objective,
                    include_config=True,
                )


def get_output_data(data_path=DATA_PATH):
    all_derived_outputs = {}
    all_outputs = {}
    for country in OPTI_REGIONS:
        country_dir_path = os.path.join(data_path, country)
        db_path = [x[0] for x in os.walk(country_dir_path)][1]
        all_derived_outputs[country] = load_derived_output_tables(db_path)[0]
        all_outputs[country] = load_output_tables(db_path)[0]

    return all_derived_outputs, all_outputs


def load_output_tables(calib_dirpath: str):
    output_tables = []
    for db_path in find_db_paths(calib_dirpath):
        db = Database(db_path)
        df = db.query("outputs")
        output_tables.append(df)

    return output_tables


def apply_scenario_mask(derived_output_df, mode, duration, objective):
    if mode is None:  # this is the baseline scenario
        sc_idx = 0
    else:
        sc_idx = get_scenario_mapping_reverse(mode, duration, objective)
    mask = derived_output_df["scenario"] == sc_idx
    return derived_output_df[mask]


def plot_stacked_outputs_by_stratum(
    base_output_df,
    sc_output_df,
    output_name: str,
    stratify_by: str,
    axis=None,
    duration=DURATIONS[0],
):

    base_last_time = list(base_output_df.times).index(PHASE_2_START_TIME)
    times_base = list(base_output_df.times)[: base_last_time - 1]
    times_sc = list(sc_output_df.times)
    times = times_base + times_sc

    if output_name == "recovered":
        base_compartments_df = base_output_df.drop(["chain", "run", "scenario", "times"], axis=1)
        sc_compartments_df = sc_output_df.drop(["chain", "run", "scenario", "times"], axis=1)

    legend = []

    running_total = [0.0] * len(times)

    blues = sns.color_palette("Blues_r", 4)
    reds = sns.color_palette("Oranges_r", 4)
    greens = sns.color_palette("BuGn_r", 4)
    purples = sns.cubehelix_palette(4)
    purples[0] = "pink"

    # mark Phase 2 in the background:
    rect = patches.Rectangle(
        (PHASE_2_START_TIME, 0),
        PHASE_2_DURATION[duration],
        1.0e9,
        linewidth=0,
        facecolor="gold",
        alpha=0.2,
    )
    rect.set_zorder(1)

    # Add the patch to the Axes
    axis.add_patch(rect)

    strata_colors = blues + reds + greens + purples

    strata = [str(5 * i) for i in range(16)]
    for color_idx, s in enumerate(strata):
        group_name = str(int(5.0 * color_idx))
        if color_idx < 15:
            group_name += "-" + str(int(5.0 * color_idx) + 4)
        else:
            group_name += "+"
        stratum_name = stratify_by + "_" + s

        if output_name == "infection_deaths":
            relevant_output_names = [
                c
                for c in list(base_output_df.columns)
                if stratum_name in c.split("X") and output_name in c
            ]
        else:
            relevant_output_names = [f"{output_name}X{stratum_name}"]

        values_0 = [0] * len(times_base)
        values_1 = [0] * len(times_sc)
        for out in relevant_output_names:
            values_0 = [
                v + d for (v, d) in zip(values_0, list(base_output_df[out])[: base_last_time - 1])
            ]
            values_1 = [v + d for (v, d) in zip(values_1, list(sc_output_df[out]))]

        if output_name == "recovered":
            deno_0 = list(np.sum(base_compartments_df, axis=1))[: base_last_time - 1]
            values_0 = [100 * v / d for (v, d) in zip(values_0, deno_0)]
            deno_1 = list(np.sum(sc_compartments_df, axis=1))
            values_1 = [100 * v / d for (v, d) in zip(values_1, deno_1)]

        new_running_total = [
            r + v for (r, v) in zip(running_total, list(values_0) + list(values_1))
        ]

        axis.fill_between(
            times,
            running_total,
            new_running_total,
            color=strata_colors[color_idx],
            label=group_name,
            zorder=2,
            alpha=1.0,
        )
        legend.append(stratum_name)
        running_total = new_running_total

    max_val = max(running_total)
    axis.set_ylim((0, 1.1 * max_val))

    # axis.axvline(x=phase_2_start, linewidth=.8, dashes=[6, 4], color='black')
    # axis.axvline(x=phase_2_end[config],linewidth=.8, dashes=[6, 4], color='black')

    phase_2_end_date = {DURATIONS[0]: "1 Apr 21", DURATIONS[1]: "1 Oct 21"}

    xticks = [
        61,
        PHASE_2_START_TIME,
        PHASE_2_START_TIME + PHASE_2_DURATION[duration],
        PHASE_2_START_TIME + PHASE_2_DURATION[duration] + 90,
    ]
    xlabs = ["1 Mar 2020", "1 Oct 20", phase_2_end_date[duration]]

    axis.set_xlim((30, PHASE_2_START_TIME + PHASE_2_DURATION[duration] + 90))
    axis.set_xticks(xticks)
    axis.set_xticklabels(xlabs, fontsize=12)

    for tick in axis.yaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    # axis.xaxis.get_major_ticks().label.set_fontsize(12)

    ylab = {
        "recovered": "% recovered",
        "incidence": "new diseased individuals",
        "infection_deaths": "number of deaths",
    }
    # axis.set_ylabel(ylab[compartment_name], fontsize=14)

    handles, labels = axis.get_legend_handles_labels()
    # axis.legend(reversed(handles), reversed(labels), bbox_to_anchor=(1.4, 1.1), title='Age:')

    return handles, labels


def plot_multicountry_rainbow(
    all_derived_outputs, all_outputs, mode, duration, objective, include_config=False
):
    fig_h = 20 if not include_config else 21
    heights = [1, 6, 6, 6, 6, 6, 6] if not include_config else [2, 1, 6, 6, 6, 6, 6, 6]

    fig = pyplot.figure(constrained_layout=True, figsize=(20, fig_h))  # (w, h)

    widths = [1, 6, 6, 6, 2]
    spec = fig.add_gridspec(ncols=5, nrows=len(heights), width_ratios=widths, height_ratios=heights)

    output_names = ["incidence", "infection_deaths", "recovered"]
    output_titles = ["Daily disease incidence", "Daily deaths", "Percentage recovered"]

    countries = ["belgium", "france", "italy", "spain", "sweden", "united-kingdom"]
    country_names = [c.title() for c in countries]
    country_names[-1] = "United Kingdom"

    text_size = 23

    for i, country in enumerate(countries):
        base_derived_output_df = apply_scenario_mask(all_derived_outputs[country], None, None, None)
        sc_derived_output_df = apply_scenario_mask(
            all_derived_outputs[country], mode, duration, objective
        )
        base_output_df = apply_scenario_mask(all_outputs[country], None, None, None)
        sc_output_df = apply_scenario_mask(all_outputs[country], mode, duration, objective)

        i_grid = i + 2 if include_config else i + 1
        for j, output_name in enumerate(output_names):
            ax = fig.add_subplot(spec[i_grid, j + 1])

            if output_name == "recovered":
                h, l = plot_stacked_outputs_by_stratum(
                    base_output_df,
                    sc_output_df,
                    output_name,
                    "agegroup",
                    axis=ax,
                    duration=duration,
                )
            else:
                h, l = plot_stacked_outputs_by_stratum(
                    base_derived_output_df,
                    sc_derived_output_df,
                    output_name,
                    "agegroup",
                    axis=ax,
                    duration=duration,
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
        ax.legend(
            reversed(h),
            reversed(l),
            title="Age:",
            fontsize=15,
            title_fontsize=text_size,
            labelspacing=1.0,
            loc="center",
        )  # bbox_to_anchor=(1.4, 1.1),
        ax.axis("off")

    if include_config:
        ax = fig.add_subplot(spec[0, :])
        obj_name = "deaths" if objective == "deaths" else "years of life lost"
        config_name = "6-month" if duration == DURATIONS[0] else "12-month"

        config_label = (
            "Optimisation by "
            + mode[3:]
            + " minimising "
            + obj_name
            + " with "
            + config_name
            + " mitigation"
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

    pyplot.rcParams["font.family"] = "Times New Roman"

    filename = f"rainbow_{mode}_{duration}_{objective}"
    png_path = os.path.join(FIGURE_PATH, f"{filename}.png")
    pdf_path = os.path.join(FIGURE_PATH, f"{filename}.pdf")
    pyplot.savefig(png_path, dpi=300)
    pyplot.savefig(pdf_path)


if __name__ == "__main__":
    main()
