import streamlit as st
from autumn import plots
from dash.dashboards.calibration.plots import get_uncertainty_data
from math import ceil
import matplotlib.pyplot as pyplot

PLOT_FUNCS = {}


def test_only(plotter, calib_dir_path, mcmc_tables, mcmc_params, targets, app_name, region_name):
    """
    Code taken directly from the fit calibration file at this stage.
    """

    chosen_output = "incidence"

    # Set up interface
    is_logscale = st.sidebar.checkbox("Log scale")

    max_n_col = 2
    n_panels = len(region_name)
    n_cols = min(max_n_col, n_panels)
    n_rows = ceil(n_panels / max_n_col)
    fig = pyplot.figure()
    i_col = 0
    i_row = 0
    for i_region in range(1, len(region_name) + 1):
        i_targets = targets[i_region]
        i_mcmc_tables = mcmc_tables[i_region]
        i_calib_dir_path = calib_dir_path[i_region]

        # Get data for plotting
        outputs = get_uncertainty_data(i_calib_dir_path, i_mcmc_tables, chosen_output, 0)

        # Call main plotting function
        axis = fig.add_subplot(1, 2, i_region)

        plots.calibration.plots.plot_calibration(
            axis,
            chosen_output,
            outputs,
            i_targets,
            is_logscale
        )

        filename = f"calibration-fit-{chosen_output}"
        title_text = f"Calibration fit for {chosen_output}"
        plotter[1].save_figure(fig, filename=filename, title_text=title_text)

        i_col += 1
        if i_col == max_n_col:
            i_col = 0
            i_row += 1


PLOT_FUNCS["Test only"] = test_only
