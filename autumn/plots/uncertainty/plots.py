"""
Plotting projection uncertainty.
"""
import logging
from datetime import datetime


from autumn.plots.plotter import Plotter
from autumn.plots.calibration.plots import _plot_targets_to_axis

logger = logging.getLogger(__name__)


def plot_timeseries_with_uncertainty(
    plotter: Plotter, output_name: str, scenario: int, quantiles: dict, times: list, targets,
):
    fig, axis, _, _, _ = plotter.get_figure()
    title = plotter.get_plot_title(output_name)
    # Plot quantiles
    colors = ["lightsteelblue", "cornflowerblue", "royalblue"]
    q_keys = sorted([float(k) for k in quantiles.keys()])
    num_quantiles = len(q_keys)
    half_length = num_quantiles // 2
    for i in range(half_length):
        color = colors[i]
        start_key = q_keys[i]
        end_key = q_keys[-(i + 1)]
        axis.fill_between(times, quantiles[start_key], quantiles[end_key], facecolor=color)

    if num_quantiles % 2:
        q_key = q_keys[half_length]
        axis.plot(times, quantiles[q_key], color="navy")

    # Add plot targets
    output_config = {"values": [], "times": []}
    for t in targets.values():
        if t["output_key"] == output_name:
            output_config = t

    values = output_config["values"]
    times = output_config["times"]
    _plot_targets_to_axis(axis, values, times, on_uncertainty_plot=True)

    axis.set_xlabel("time")
    axis.set_ylabel(output_name)
    scenario_title = "baseline scenario" if scenario == 0 else f"scenario #{scenario}"
    plotter.save_figure(
        fig,
        filename=f"uncertainty-{output_name}-{scenario}",
        title_text=f"{title} for {scenario_title}",
    )
