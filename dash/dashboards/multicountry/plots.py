import streamlit as st
from autumn.plots.calibration.plots import find_shortest_chain_length
from autumn import plots
from dash.dashboards.calibration.plots import get_uncertainty_data

PLOT_FUNCS = {}


def test_only(plotter, calib_dir_path, mcmc_tables, mcmc_params, targets, app_name, region_name):
    """
    Code taken directly from the fit calibration file at this stage.
    """

    i_plotter = plotter[1]
    i_targets = targets[1]
    i_mcmc_tables = mcmc_tables[1]
    i_calib_dir_path = calib_dir_path[1]

    # Set up interface
    available_outputs = [o["output_key"] for o in i_targets.values()]
    chain_length = find_shortest_chain_length(i_mcmc_tables)
    burn_in = st.sidebar.slider("Burn in (select 0 for default behaviour of discarding first half)", 0,
                                chain_length, 0)
    chosen_output = st.sidebar.selectbox("Select calibration target", available_outputs)
    is_logscale = st.sidebar.checkbox("Log scale")

    # Get data for plotting
    outputs = get_uncertainty_data(i_calib_dir_path, i_mcmc_tables, chosen_output, burn_in)

    # Call main plotting function
    plots.calibration.plots.plot_calibration_fit(
        i_plotter,
        chosen_output,
        outputs,
        i_targets,
        is_logscale
    )


PLOT_FUNCS["Test only"] = test_only
