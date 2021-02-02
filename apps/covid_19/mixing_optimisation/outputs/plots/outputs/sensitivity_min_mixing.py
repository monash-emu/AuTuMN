import yaml
import os
from matplotlib import pyplot

from settings import BASE_PATH

from apps.covid_19.mixing_optimisation.mixing_opti import DURATIONS

FIGURE_PATH = os.path.join(
    BASE_PATH,
    "apps",
    "covid_19",
    "mixing_optimisation",
    "outputs",
    "plots",
    "outputs",
    "figures",
    "sensitivity_min_mixing",
)


def main():
    plot_multicountry_min_mixing_sensitivity()


def plot_sensitivity_min_mixing(results, country, duration, axis):

    x_vals = list(results[country][duration].keys())
    x_vals.sort()

    y_vals_deaths = [results[country][duration][x]["d"] for x in x_vals]

    y_vals_yoll = [results[country][duration][x]["yoll"] for x in x_vals]

    colors = ["purple", "dodgerblue"]
    axis.set_xlabel("minimum mixing factor", fontsize=15)

    axis.plot(x_vals, y_vals_deaths, color=colors[0])
    ax2 = axis.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(x_vals, y_vals_yoll, color=colors[1])

    axis.set_ylabel("deaths", color=colors[0], fontsize=15)
    axis.tick_params(axis="y", labelcolor=colors[0])
    axis.set_ylim([0.0, max(y_vals_deaths) * 1.3])
    ax2.set_ylabel("years of life lost", color=colors[1], fontsize=15)
    ax2.tick_params(axis="y", labelcolor=colors[1])
    ax2.set_ylim([0.0, max(y_vals_yoll) * 1.05])


def plot_multicountry_min_mixing_sensitivity():
    path = os.path.join(
        BASE_PATH,
        "apps",
        "covid_19",
        "mixing_optimisation",
        "sensitivity_analyses",
        "min_mixing_results.yml",
    )

    with open(path, "r") as yaml_file:
        results = yaml.safe_load(yaml_file)

    heights = [1, 6, 6, 6, 6, 6, 6]
    pyplot.rcParams["font.family"] = "Times New Roman"
    pyplot.style.use("default")
    fig = pyplot.figure(constrained_layout=True, figsize=(15, 20))  # (w, h)

    widths = [1, 6, 6]
    spec = fig.add_gridspec(
        ncols=3, nrows=len(heights), width_ratios=widths, height_ratios=heights, wspace=1.0
    )

    duration_titles = ["Six-month mitigation", "Twelve-month mitigation"]

    countries = ["belgium", "france", "italy", "spain", "sweden", "united-kingdom"]
    country_names = [c.title() for c in countries]
    country_names[-1] = "United Kingdom"

    text_size = 23

    for i, country in enumerate(countries):
        i_grid = i + 1
        for j, duration in enumerate(DURATIONS):
            ax = fig.add_subplot(spec[i_grid, j + 1])
            plot_sensitivity_min_mixing(results, country, duration, ax)
            if i == 0:
                i_grid_output_labs = 0
                ax = fig.add_subplot(spec[i_grid_output_labs, j + 1])
                ax.text(
                    0.5,
                    0.5,
                    duration_titles[j],
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

    filename = "sensitivity_min_mixing"
    png_path = os.path.join(FIGURE_PATH, f"{filename}.png")
    pdf_path = os.path.join(FIGURE_PATH, f"{filename}.pdf")

    pyplot.savefig(png_path, dpi=300)
    pyplot.savefig(pdf_path)


if __name__ == "__main__":
    main()
