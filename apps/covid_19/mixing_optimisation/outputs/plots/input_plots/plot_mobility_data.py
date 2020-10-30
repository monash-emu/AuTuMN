import os
from matplotlib import pyplot
import matplotlib as mpl
import seaborn as sns

from autumn.constants import BASE_PATH
from apps.covid_19.mixing_optimisation.constants import OPTI_REGIONS

FIGURE_PATH = os.path.join(BASE_PATH, "apps", "covid_19", "mixing_optimisation",
                           "outputs", "plots", "input_plots", "figures")


def main():
    # Reset pyplot style
    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.pyplot.style.use("ggplot")

    mobility_data_functions = get_mobility_data()
    plot_multicountry_mobility(mobility_data_functions)


def get_mobility_data():
    mobility_data_functions = {}
    for country in OPTI_REGIONS:
        mobility_data_functions[country] = {}
        for loc_key in ("other_locations", "school", "work"):
            mobility_func = lambda t: .5  # FIXME: need to load the time-variant function here
            mobility_data_functions[country][loc_key] = mobility_func

    return mobility_data_functions


def plot_mobility(mobility_data_functions, country, loc_key, axis):
    t_max = 300

    times = list(range(t_max))
    values = [mobility_data_functions[country][loc_key](t) for t in times]

    my_color = sns.cubehelix_palette(4)[-1]

    axis.plot(times, values, linewidth=2, color=my_color)

    xticks = [92, 153, 214, 275]
    xlabs = ["1 Apr 2020", "1 Jun 2020", "1 Aug 2020", "1 Oct 2020"]

    axis.set_xlim((60, t_max))
    axis.set_xticks(xticks)
    axis.set_xticklabels(xlabs, fontsize=12)

    axis.set_ylim((0., 1.1))


def plot_multicountry_mobility(mobility_data_functions):
    fig = pyplot.figure(constrained_layout=True, figsize=(18, 20))  # (w, h)
    widths = [1, 6, 6, 6]
    heights = [1, 6, 6, 6, 6, 6, 6]
    spec = fig.add_gridspec(ncols=4, nrows=7, width_ratios=widths,
                            height_ratios=heights)

    location_names = {
        "other_locations": "other locations",
        "school": "schools",
        "work": "workplaces",
    }

    countries = ['belgium', 'france', 'italy', 'spain', 'sweden', 'united-kingdom']
    country_names = [c.title() for c in countries]
    country_names[-1] = "United Kingdom"

    text_size = 23

    for i, country in enumerate(countries):
        for j, loc_key in enumerate(["other_locations", "work", "school"]):
            ax = fig.add_subplot(spec[i+1, j + 1])
            plot_mobility(mobility_data_functions, country, loc_key, axis=ax)
            if i == 0:
                ax = fig.add_subplot(spec[0, j+1])
                ax.text(0.5, 0.5, location_names[loc_key], fontsize=text_size, horizontalalignment='center', verticalalignment='center')
                ax.axis("off")

        ax = fig.add_subplot(spec[i+1, 0])
        ax.text(0.5, 0.5, country_names[i], rotation=90, fontsize=text_size, horizontalalignment='center', verticalalignment='center')
        ax.axis("off")

    pyplot.rcParams["font.family"] = "Times New Roman"

    filename = "mobility_inputs"
    png_path = os.path.join(FIGURE_PATH, f"{filename}.png")
    pdf_path = os.path.join(FIGURE_PATH, f"{filename}.pdf")

    pyplot.savefig(png_path, dpi=300)
    pyplot.savefig(pdf_path)


if __name__ == "__main__":
    main()
