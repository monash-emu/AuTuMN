from matplotlib import pyplot
import matplotlib.patches as patches
import os
import yaml

from apps.covid_19.mixing_optimisation.constants import OPTI_REGIONS
from apps.covid_19.mixing_optimisation.mixing_opti import MODES, DURATIONS, OBJECTIVES, run_root_model, objective_function
from apps.covid_19.mixing_optimisation.write_scenarios import read_opti_outputs, read_decision_vars
from autumn.constants import BASE_PATH


FIGURE_PATH = os.path.join(BASE_PATH, "apps", "covid_19", "mixing_optimisation",
                           "outputs", "plots", "outputs", "figures", "optimal_plan")


def main():
    opti_output_filename = "dummy_vars_for_test.csv"
    opti_outputs_df = read_opti_outputs(opti_output_filename)
    all_results = {}
    for country in OPTI_REGIONS:
        all_results[country] = {}
        for mode in MODES:
            all_results[country][mode] = {}
            for duration in DURATIONS:
                all_results[country][mode][duration] = {}
                for objective in OBJECTIVES:

                    all_results[country][mode][duration][objective] = read_decision_vars(
                        opti_outputs_df, country, mode, duration, objective
                    )

    for mode in MODES:
        plot_multicountry_optimal_plan(all_results, mode)


def plot_optimal_plan(all_results, duration, country, mode, ax):

    colours = {
        'deaths': 'lightcoral',
        'yoll': 'skyblue',
    }
    n_vars = {
        'by_age': 16,
        'by_location': 3,
    }
    bar_width = .32

    ax.grid(linewidth=.5, zorder=0, linestyle="dotted")
    ax.set_axisbelow(True)

    # Load sensitivity outputs
    sensi_outputs = {}
    dir_path = os.path.join(
        BASE_PATH, "apps", "covid_19", "mixing_optimisation", "optimised_variables", "optimal_plan_sensitivity"
    )
    directions = ["down", "up"]
    for objective in ('deaths', 'yoll'):
        sensi_outputs[objective] = {}
        for direction in directions:
            path = os.path.join(
                dir_path, country + "_" + mode + "_" + duration + "_" + objective + "_" + direction + ".yml"
            )
            try:
                with open(path, "r") as yaml_file:
                    sensi_outputs[objective][direction] = yaml.safe_load(yaml_file)
            except:
                sensi_outputs[objective][direction] = [0.] * n_vars[mode]

    ymax = 0.
    for i_age in range(n_vars[mode]):
        x_pos = i_age + 1.
        delta_xpos = -1
        for objective in ('deaths', 'yoll'):
            if all_results[country][mode][duration][objective] is None:
                value = .5
            else:
                value = all_results[country][mode][duration][objective][i_age]

            rect = patches.Rectangle((x_pos + delta_xpos * bar_width, 0.), bar_width, value, linewidth=.8,
                              facecolor=colours[objective], edgecolor='black', zorder=1)
            ax.add_patch(rect)

            for direction in directions:
                arrow_length = sensi_outputs[objective][direction][i_age]
                if direction == "down":
                    arrow_length *= -1.
                _x = x_pos + delta_xpos * bar_width + .5 * bar_width
                ax.plot((_x, _x), (value, value + arrow_length), color='black', linewidth=1.5, zorder=3)

                ymax = max([ymax, value + arrow_length])

            delta_xpos = 0

    if mode == "by_age":
        # X axis settings
        major_ticks = [i + .5 for i in range(1, 16)]
        minor_ticks = range(1, 17)
        age_names = [ str(i*5) + "-" + str(i*5 + 4) for i in range(16)]
        age_names[-1] = "75+"

        ax.set_xticklabels(age_names, minor=True, rotation=45, fontsize=11)

        # Y axis settinds
        ylab = "Age-specific mixing factor"
    else:
        major_ticks = [1.5, 2.5]
        minor_ticks = [1, 2, 3]

        ylab = "Relative contact rate"
        ax.set_xticklabels(("other locations", "schools", "workplaces"), minor=True, fontsize=13)

    # ax.axhline(y=0.1, color='dimgrey', dashes=(1.2, 1.2), zorder=-10)
    # ax.axhline(y=1., color='dimgrey', dashes=(1.2, 1.2), linewidth=1., zorder=-10)
    rect = patches.Rectangle((-10, 0.1), 100, 0.9, facecolor='linen', zorder=-10)
    ax.add_patch(rect)

    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)

    ax.set_xticklabels("", major=True)
    ax.tick_params(axis="x", which="minor", length=0)
    ax.tick_params(axis="x", which="major", length=4)
    ax.set_xlim((0.5, n_vars[mode] + .5))

    ax.set_ylim((0, max((ymax, 1.09))))

    if duration == DURATIONS[0]:
        ax.set_ylabel(ylab, fontsize=14)


def plot_multicountry_optimal_plan(all_results, mode):
    pyplot.style.use("default")

    fig_width = {
        'by_age': 20,
        'by_location': 12
    }
    fig = pyplot.figure(constrained_layout=True, figsize=(fig_width[mode], 20))  # (w, h)
    pyplot.rcParams["font.family"] = "Times New Roman"

    widths = [1, 8, 8]
    heights = [1, 4, 4, 4, 4, 4, 4]
    spec = fig.add_gridspec(ncols=3, nrows=7, width_ratios=widths,
                            height_ratios=heights, hspace=0)
    text_size = 23

    countries = ['belgium', 'france', 'italy', 'spain', 'sweden', 'united-kingdom']
    country_names = [c.title() for c in countries]
    country_names[-1] = "United Kingdom"

    duration_names = ("Six-month mitigation phase", "Twelve-month mitigation phase")

    for j, duration in enumerate(DURATIONS):
        for i, country in enumerate(countries):
            ax = fig.add_subplot(spec[i+1, j + 1])
            plot_optimal_plan(all_results, duration, country, mode, ax)

            if j == 0:
                ax = fig.add_subplot(spec[i+1, 0])
                ax.text(0.8, 0.5, country_names[i], rotation=90, fontsize=text_size,
                        horizontalalignment='center', verticalalignment='center', fontweight='normal')
                ax.axis("off")

        ax = fig.add_subplot(spec[0, j + 1])
        ax.text(0.5, 0.5, duration_names[j], fontsize=text_size, horizontalalignment='center', verticalalignment='center',
                fontweight='normal')
        ax.axis("off")

    filename = "optimal_plan_" + mode
    png_path = os.path.join(FIGURE_PATH, f"{filename}.png")
    pdf_path = os.path.join(FIGURE_PATH, f"{filename}.pdf")

    pyplot.savefig(png_path, dpi=300)
    pyplot.savefig(pdf_path)


if __name__ == "__main__":
    main()
