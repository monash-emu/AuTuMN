import streamlit as st
from autumn import plots
from dash.dashboards.calibration.plots import get_uncertainty_data
from math import ceil
import matplotlib.pyplot as pyplot

PLOT_FUNCS = {}


def multi_country_fit(plotter, calib_dir_path, mcmc_tables, mcmc_params, targets, app_name, region_name):
    """
    Code taken directly from the fit calibration file at this stage.
    """

    chosen_output = "incidence"

    # Set up interface
    is_logscale = st.sidebar.checkbox("Log scale")
    i_plotter = plotter[1]

    fig, axes, _, n_rows, n_cols, indices = \
        i_plotter.get_figure(len(region_name), share_xaxis=True)

    for i_region in range(n_rows * n_cols):
        if i_region < len(region_name):

            i_targets = targets[i_region]
            i_mcmc_tables = mcmc_tables[i_region]
            i_calib_dir_path = calib_dir_path[i_region]

            # Get data for plotting
            outputs = get_uncertainty_data(i_calib_dir_path, i_mcmc_tables, chosen_output, 0)

            # Call main plotting function
            plots.calibration.plots.plot_calibration(
                axes[indices[i_region][0], indices[i_region][1]],
                chosen_output,
                outputs,
                i_targets,
                is_logscale
            )
        else:
            axes[indices[i_region][0], indices[i_region][1]].axis("off")

    filename = f"calibration-fit-{chosen_output}"
    title_text = f"Calibration fit for {chosen_output}"
    i_plotter.save_figure(fig, filename=filename, title_text=title_text)



PLOT_FUNCS["Test only"] = multi_country_fit
