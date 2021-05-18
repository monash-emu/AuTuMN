import os

from matplotlib import pyplot

from autumn.projects.covid_19.mixing_optimisation.constants import (
    PHASE_2_DURATION,
    PHASE_2_START_TIME,
)
from autumn.projects.covid_19.mixing_optimisation.mixing_opti import DURATIONS, MODES, OBJECTIVES
from autumn.projects.covid_19.mixing_optimisation.outputs.plots.outputs.rainbows import (
    apply_scenario_mask,
    get_output_data,
)
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
    "attack_rate_by_age",
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
    all_derived_outputs, _ = get_output_data()

    for mode in MODES:
        plot_multicountry_attack_rates_by_age(all_derived_outputs, mode)


def plot_attack_rates_by_age(all_derived_outputs, country, mode, duration, objective, ax):
    derived_outputs = apply_scenario_mask(all_derived_outputs[country], mode, duration, objective)

    times = list(derived_outputs.times)
    phase_2_end_time = PHASE_2_START_TIME + PHASE_2_DURATION[duration]
    time_index = times.index(phase_2_end_time)
    ind = range(16)

    heights = []
    ax.grid(linewidth=0.5, zorder=0, linestyle="dotted", axis="y")
    ax.set_axisbelow(True)

    for age_index in range(16):
        age_group_name = "agegroup_" + str(int(5.0 * age_index))
        perc_reco = (
            100 * list(derived_outputs[f"proportion_seropositiveX{age_group_name}"])[time_index]
        )
        heights.append(perc_reco)

    ax.bar(ind, heights, width=0.5, color="mediumorchid")

    xlabs = [str(i * 5) + "-" + str(i * 5 + 4) for i in range(16)]
    xlabs[-1] = "75+"
    ax.set_xticks(ind)
    ax.set_xticklabels(xlabs, rotation=45, fontsize=10)

    ax.set_ylim((0, 100.0))

    if duration == DURATIONS[0] and objective == OBJECTIVES[0]:
        ax.set_ylabel("% recovered", fontsize=16)


def plot_multicountry_attack_rates_by_age(all_derived_outputs, mode="by_age"):

    pyplot.style.use("default")

    fig = pyplot.figure(constrained_layout=True, figsize=(25, 20))  # (w, h)
    pyplot.rcParams["font.family"] = "Times New Roman"

    widths = [1, 4, 4, 4, 4]
    heights = [1, 4, 4, 4, 4, 4, 4]
    spec = fig.add_gridspec(
        ncols=5, nrows=7, width_ratios=widths, height_ratios=heights, hspace=0, wspace=0
    )
    text_size = 23

    countries = ["belgium", "france", "italy", "spain", "sweden", "united-kingdom"]
    country_names = [c.title() for c in countries]
    country_names[-1] = "United Kingdom"

    config_names = ("Six-month mitigation phase", "Twelve-month mitigation phase")
    objective_names = ("minimising deaths", "minimising YLLs")

    col_index = 0
    for j, duration in enumerate(DURATIONS):
        for h, objective in enumerate(OBJECTIVES):
            col_index += 1
            for i, country in enumerate(countries):
                ax = fig.add_subplot(spec[i + 1, col_index])
                plot_attack_rates_by_age(
                    all_derived_outputs, country, mode, duration, objective, ax
                )

                if j + h == 0:
                    ax = fig.add_subplot(spec[i + 1, 0])
                    ax.text(
                        0.8,
                        0.5,
                        country_names[i],
                        rotation=90,
                        fontsize=text_size,
                        horizontalalignment="center",
                        verticalalignment="center",
                        fontweight="normal",
                    )
                    ax.axis("off")

        ax = fig.add_subplot(spec[0, 2 * j + 1 : 2 * j + 3])
        ax.text(
            0.5,
            0.8,
            config_names[j],
            fontsize=text_size,
            horizontalalignment="center",
            verticalalignment="center",
            fontweight="normal",
        )

        x_pos_labels = [0.25, 0.75]
        for h in [0, 1]:
            ax.text(
                x_pos_labels[h],
                0.0,
                objective_names[h],
                fontsize=text_size,
                horizontalalignment="center",
                verticalalignment="center",
                fontweight="normal",
            )
        ax.axis("off")

    filename = f"age_specific_attack_rates_{mode}"
    png_path = os.path.join(FIGURE_PATH, f"{filename}.png")
    pdf_path = os.path.join(FIGURE_PATH, f"{filename}.pdf")
    pyplot.savefig(png_path, dpi=300)
    pyplot.savefig(pdf_path)


if __name__ == "__main__":
    main()
