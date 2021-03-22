from math import ceil

import matplotlib.pyplot as pyplot
import streamlit as st

from autumn import plots
from dash.dashboards.calibration_results.plots import get_uncertainty_data

PLOT_FUNCS = {}


def multi_country_fit(
    plotter, calib_dir_path, mcmc_tables, mcmc_params, targets, app_name, region_name
):
    """
    Code taken directly from the fit calibration file at this stage.
    """

    # Set up interface
    available_outputs = [o["output_key"] for o in targets[0].values()]
    chosen_output = st.sidebar.selectbox("Select calibration target", available_outputs)
    outputs = []

    fig, axes, _, n_rows, n_cols, indices = plotter.get_figure(len(region_name), share_xaxis=True)

    # Get data for plotting
    for i_region in range(len(region_name)):
        outputs.append(
            get_uncertainty_data(calib_dir_path[i_region], mcmc_tables[i_region], chosen_output, 0)
        )

    for i_region in range(n_rows * n_cols):
        axis = axes[indices[i_region][0], indices[i_region][1]]

        if i_region < len(region_name):
            # Call main plotting function
            plots.calibration.plots.plot_calibration(
                axis, chosen_output, outputs[i_region], targets[i_region], False
            )
            axis.set_title(region_name[i_region], fontsize=10)

        else:
            axis.axis("off")

    filename = f"calibration-fit-{chosen_output}"
    title_text = f"Calibration fit for {chosen_output}"
    plotter.save_figure(fig, filename=filename, title_text=title_text)


PLOT_FUNCS["Multi-country fit"] = multi_country_fit
