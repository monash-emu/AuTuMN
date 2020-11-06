import os
from matplotlib import pyplot
from matplotlib.pyplot import figure

from autumn.curve.tanh import tanh_based_scaleup
from autumn.constants import BASE_PATH


FIGURE_PATH = os.path.join(BASE_PATH, "apps", "covid_19", "mixing_optimisation",
                           "outputs", "plots", "input_plots", "figures")

PARAMS = {
    'mobility.microdistancing.behaviour.parameters.c': 61,
    'mobility.microdistancing.behaviour.parameters.upper_asymptote': 0.4,
    'mobility.microdistancing.behaviour_adjuster.parameters.c': 183,
    'mobility.microdistancing.behaviour_adjuster.parameters.sigma': 0.75,
}


def main():
    make_microdistancing_plot()


def make_microdistancing_plot(params=PARAMS):

    micro_emergence_func = tanh_based_scaleup(
        b=0.05,
        c=params['mobility.microdistancing.behaviour.parameters.c'],
        sigma=0,
        upper_asymptote=params['mobility.microdistancing.behaviour.parameters.upper_asymptote'],
    )

    micro_wane_func = tanh_based_scaleup(
        b=-0.05,
        c=params['mobility.microdistancing.behaviour_adjuster.parameters.c'],
        sigma=params['mobility.microdistancing.behaviour_adjuster.parameters.sigma'],
        upper_asymptote=1,
    )

    md_func = lambda t: 1. - micro_emergence_func(t) * micro_wane_func(t)
    times = list(range(-30, 350))
    values = [md_func(t) for t in times]

    x_ticks = [61, 183, 306]
    x_tick_labels = ['1 Mar 2020', "1 Jul 2020", "1 Nov 2020"]

    figure(figsize=(6, 4))
    pyplot.style.use('ggplot')

    pyplot.plot(times, values, '-', color='cornflowerblue', lw=1.7)

    pyplot.ylabel('micro-distancing multiplier')

    pyplot.ylim((0, 1.05))
    pyplot.tight_layout()

    pyplot.xticks(x_ticks, x_tick_labels)

    filename = "microdistancing"

    png_path = os.path.join(FIGURE_PATH, f"{filename}.png")
    pdf_path = os.path.join(FIGURE_PATH, f"{filename}.pdf")

    pyplot.savefig(png_path, dpi=300)
    pyplot.savefig(pdf_path)


if __name__ == "__main__":
    main()

