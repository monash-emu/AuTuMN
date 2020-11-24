from typing import Dict

import numpy as np
from matplotlib import pyplot


def plot_timeseries(title: str, times: np.ndarray, values: Dict[str, np.ndarray]):
    pyplot.style.use("ggplot")
    fig = pyplot.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(title)
    ax.set_xlabel("times")
    legend = []
    for plot_name, plot_vals in values.items():
        ax.plot(times, plot_vals)
        legend.append(plot_name)

    ax.legend(legend)
    pyplot.show()
