from matplotlib import pyplot
import matplotlib.patches as patches
import os
import yaml
import matplotlib as mpl
import seaborn as sns
import numpy as np

from apps.covid_19.mixing_optimisation.constants import PHASE_2_START_TIME, PHASE_2_DURATION, DURATION_PHASES_2_AND_3
from apps.covid_19.mixing_optimisation.mixing_opti import MODES, DURATIONS, OBJECTIVES
from autumn.constants import BASE_PATH


FIGURE_PATH = os.path.join(BASE_PATH, "apps", "covid_19", "mixing_optimisation",
                           "outputs", "plots", "outputs", "figures", "rainbows")


def main():

    for mode in MODES:
        for duration in DURATIONS:
            for objective in OBJECTIVES:
                country_scenarios = None
                plot_multicountry_rainbow(country_scenarios, mode, duration, objective, include_config=False)


def plot_stacked_compartments_by_stratum(scenarios, compartment_name: str,
                                         stratify_by: str, multicountry=False, axis=None, duration=DURATIONS[0]):
    models = [sc.model for sc in scenarios]
    times = (models[0].times[:-1] + models[1].times)

    legend = []
    strata = models[0].all_stratifications[stratify_by]

    running_total = [0.] * len(times)

    blues = sns.color_palette("Blues_r", 4)
    reds = sns.color_palette("Oranges_r", 4)
    greens = sns.color_palette("BuGn_r", 4)
    purples = sns.cubehelix_palette(4)
    purples[0] = 'pink'

    # mark Phase 2 in the background:
    rect = patches.Rectangle((PHASE_2_START_TIME, 0), PHASE_2_DURATION[duration], 1.e9, linewidth=0,
                              facecolor='gold', alpha=.2)
    rect.set_zorder(1)

    # Add the patch to the Axes
    axis.add_patch(rect)

    strata_colors = blues + reds + greens + purples

    for color_idx, s in enumerate(strata):
        group_name = str(int(5.*color_idx))
        if color_idx < 15:
            group_name += "-" + str(int(5.*color_idx) + 4)
        else:
            group_name += "+"
        stratum_name = stratify_by + "_" + s

        if compartment_name in [c.split("X")[0] for c in models[0].compartment_names]:  # use outputs
            comp_names = [c for c in models[0].compartment_names if stratum_name in c.split('X') and compartment_name in c]
            comp_idx = [models[0].compartment_names.index(c) for c in comp_names]
            relevant_outputs_0 = models[0].outputs[:, comp_idx]
            values_0 = np.sum(relevant_outputs_0, axis=1)

            relevant_outputs_1 = models[1].outputs[:, comp_idx]
            values_1 = np.sum(relevant_outputs_1, axis=1)

            if compartment_name == 'recovered':
                deno_0 = np.sum(models[0].outputs, axis=1)
                values_0 = [100*v / d for (v, d) in zip(values_0, deno_0)]
                deno_1 = np.sum(models[1].outputs, axis=1)
                values_1 = [100*v / d for (v, d) in zip(values_1, deno_1)]

        else:  # use derived outputs
            relevant_output_names = [c for c in models[0].derived_outputs if stratum_name in c.split('X') and compartment_name in c]
            values_0 = [0] * len(models[0].times)
            values_1 = [0] * len(models[1].times)
            for out in relevant_output_names:
                values_0 = [v + d for (v, d) in zip(values_0, models[0].derived_outputs[out])]
                values_1 = [v + d for (v, d) in zip(values_1, models[1].derived_outputs[out])]

        new_running_total = [r + v for (r, v) in zip(running_total, list(values_0)[:-1] + list(values_1))]

        axis.fill_between(times, running_total, new_running_total, color=strata_colors[color_idx], label=group_name,
                          zorder=2, alpha=1.)
        legend.append(stratum_name)
        running_total = new_running_total

    max_val = max(running_total)
    axis.set_ylim((0, 1.1 * max_val))

    # axis.axvline(x=phase_2_start, linewidth=.8, dashes=[6, 4], color='black')
    # axis.axvline(x=phase_2_end[config],linewidth=.8, dashes=[6, 4], color='black')

    xticks = [61, 214, 398, 366 + 214, 366 + 365]
    xlabs = ["1 Mar 2020", "1 Aug 2020", "1 Feb 2021", "1 Aug 2021", "31 Dec 2021"]

    axis.set_xlim((30, PHASE_2_START_TIME + DURATION_PHASES_2_AND_3))
    axis.set_xticks(xticks)
    axis.set_xticklabels(xlabs, fontsize=12)

    for tick in axis.yaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    # axis.xaxis.get_major_ticks().label.set_fontsize(12)

    ylab = {
        "recovered": "% recovered",
        "incidence": "new diseased individuals",
        "infection_deaths": "number of deaths"
    }
    # axis.set_ylabel(ylab[compartment_name], fontsize=14)

    handles, labels = axis.get_legend_handles_labels()
    # axis.legend(reversed(handles), reversed(labels), bbox_to_anchor=(1.4, 1.1), title='Age:')

    return handles, labels


def plot_multicountry_rainbow(country_scenarios, mode, duration, objective, include_config=False):
    fig_h = 20 if not include_config else 21
    heights = [1, 6, 6, 6, 6, 6, 6] if not include_config else [2, 1, 6, 6, 6, 6, 6, 6]

    fig = pyplot.figure(constrained_layout=True, figsize=(20, fig_h))  # (w, h)

    widths = [1, 6, 6, 6, 2]
    spec = fig.add_gridspec(ncols=5, nrows=len(heights), width_ratios=widths,
                            height_ratios=heights)

    output_names = ["incidence", "infection_deaths", "recovered"]
    output_titles = ["Daily disease incidence", "Daily deaths", "Percentage recovered"]

    countries = ['belgium', 'france', 'italy', 'spain', 'sweden', 'united-kingdom']
    country_names = [c.title() for c in countries]
    country_names[-1] = "United Kingdom"

    text_size = 23

    for i, country in enumerate(countries):
        i_grid = i + 2 if include_config else i+1
        for j, output in enumerate(output_names):
            ax = fig.add_subplot(spec[i_grid, j + 1])
            h, l = plot_stacked_compartments_by_stratum(
                None, country_scenarios[country][mode][duration][objective], output, "agegroup", multicountry=True,
                axis=ax, duration=duration
            )
            if i == 0:
                i_grid_output_labs = 1 if include_config else 0
                ax = fig.add_subplot(spec[i_grid_output_labs, j+1])
                ax.text(0.5, 0.5, output_titles[j], fontsize=text_size, horizontalalignment='center', verticalalignment='center')
                ax.axis("off")

        ax = fig.add_subplot(spec[i_grid, 0])
        ax.text(0.5, 0.5, country_names[i], rotation=90, fontsize=text_size, horizontalalignment='center', verticalalignment='center')
        ax.axis("off")

    if j == 2:
        ax = fig.add_subplot(spec[1:, 4])
        ax.legend(reversed(h), reversed(l), title='Age:', fontsize=15, title_fontsize=text_size,
                  labelspacing=1.0, loc='center')  # bbox_to_anchor=(1.4, 1.1),
        ax.axis("off")

    if include_config:
        ax = fig.add_subplot(spec[0, :])
        obj_name = 'deaths' if objective == 'deaths' else 'years of life lost'
        config_name = '6-month' if duration == DURATIONS[0] else '12-month'

        config_label = "Optimisation by " + mode[3:] + " minimising " + obj_name + " with " + config_name + " mitigation"
        ax.text(0., 0.5, config_label, fontsize=text_size, horizontalalignment='left', verticalalignment='center')
        ax.axis("off")

    pyplot.rcParams["font.family"] = "Times New Roman"

    filename = f"rainbow_{mode}_{duration}_{objective}"
    png_path = os.path.join(FIGURE_PATH, f"{filename}.png")
    pdf_path = os.path.join(FIGURE_PATH, f"{filename}.pdf")
    pyplot.savefig(png_path, dpi=300)
    pyplot.savefig(pdf_path)


if __name__ == "__main__":
    main()
