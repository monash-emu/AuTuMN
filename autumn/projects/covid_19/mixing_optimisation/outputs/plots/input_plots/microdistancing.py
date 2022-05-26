import os

from matplotlib import pyplot
from matplotlib.pyplot import figure

from autumn.model_features.curve.tanh import tanh_based_scaleup
from autumn.settings import BASE_PATH

FIGURE_PATH = os.path.join(
    BASE_PATH,
    "apps",
    "covid_19",
    "mixing_optimisation",
    "outputs",
    "plots",
    "input_plots",
    "figures",
)

PARAMS = {
    "mobility.microdistancing.behaviour.parameters.inflection_time": 61,
    "mobility.microdistancing.behaviour.parameters.end_asymptote": 0.4,
    "mobility.microdistancing.behaviour_adjuster.parameters.inflection_time": 183,
    "mobility.microdistancing.behaviour_adjuster.parameters.start_asymptote": 0.75,
}


def main():
    make_microdistancing_plot()


def make_microdistancing_plot(params=PARAMS):

    micro_emergence_func = tanh_based_scaleup(
        shape=0.05,
        inflection_time=params["mobility.microdistancing.behaviour.parameters.inflection_time"],
        start_asymptote=0,
        end_asymptote=params["mobility.microdistancing.behaviour.parameters.end_asymptote"],
    )

    micro_wane_func = tanh_based_scaleup(
        shape=-0.05,
        inflection_time=params[
            "mobility.microdistancing.behaviour_adjuster.parameters.inflection_time"
        ],
        start_asymptote=params[
            "mobility.microdistancing.behaviour_adjuster.parameters.start_asymptote"
        ],
        end_asymptote=1,
    )

    md_func = lambda t: 1.0 - micro_emergence_func(t) * micro_wane_func(t)
    times = list(range(-30, 350))
    values = [md_func(t) for t in times]

    x_ticks = [61, 183, 306]
    x_tick_labels = ["1 Mar 2020", "1 Jul 2020", "1 Nov 2020"]

    figure(figsize=(6, 4))
    pyplot.style.use("ggplot")

    pyplot.plot(times, values, "-", color="cornflowerblue", lw=1.7)

    pyplot.ylabel("micro-distancing multiplier")

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
