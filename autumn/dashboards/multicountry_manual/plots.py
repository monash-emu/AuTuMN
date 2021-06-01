from math import ceil

import matplotlib.pyplot as pyplot
import streamlit as st

from autumn.tools.plots.model.plots import _plot_outputs_to_axis, _plot_targets_to_axis
from dash.dashboards.model_results.plots import model_output_selector

PLOT_FUNCS = {}


def multi_country_manual(plotter, scenarios, targets, app_name, region_names):

    # Set up interface
    available_outputs = [o["output_key"] for o in targets[0].values()]
    chosen_output = st.sidebar.selectbox("Select calibration target", available_outputs)

    fig, axes, _, n_rows, n_cols, indices = plotter.get_figure(len(region_names), share_xaxis=True)

    for i_region in range(n_rows * n_cols):
        axis = axes[indices[i_region][0], indices[i_region][1]]

        if i_region < len(region_names):
            # plot the scenarios
            legend = []
            for idx, scenario in enumerate(reversed(scenarios[i_region])):
                color_idx = len(scenarios[i_region]) - idx - 1
                _plot_outputs_to_axis(axis, scenario, chosen_output, color_idx=color_idx, alpha=0.7)
                legend.append(scenario.name)
            # axis.legend(legend)

            # plot the targets
            target_times = targets[i_region][chosen_output]["times"]
            target_values = targets[i_region][chosen_output]["values"]
            _plot_targets_to_axis(axis, target_values, target_times)

            axis.set_title(region_names[i_region], fontsize=10)

        else:
            axis.axis("off")

    fig.set_figwidth(10)

    filename = f"multi-manual-{chosen_output}"
    title_text = f"Model outputs for {chosen_output}"
    plotter.save_figure(fig, filename=filename, title_text=title_text)


PLOT_FUNCS["Multi-country manual"] = multi_country_manual
