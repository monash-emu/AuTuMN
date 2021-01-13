"""
Streamlit web UI for plotting MCMC outputs
"""
import streamlit as st
import os

from autumn.tool_kit.params import load_targets
from autumn import db
from autumn.plots.plotter import StreamlitPlotter, FilePlotter


from dash import selectors
from .plots import PLOT_FUNCS


def run_dashboard():
    app_name, app_dirpath = "covid_19", os.path.join(os.getcwd(), "data\outputs\calibrate\covid_19")
    if not app_name:
        st.write("No calibrations have been run yet")
        return

    region_name = "victoria"
    region_dirpath = os.path.join(app_dirpath, region_name)
    if not region_name:
        st.write("No region folder found")
        return

    calib_name, calib_dirpath = selectors.calibration_run(region_dirpath, region_name)
    if not calib_name:
        st.write("No model run folder found")
        return

    # Load MCMC tables
    mcmc_tables = db.load.load_mcmc_tables(calib_dirpath)
    mcmc_params = db.load.load_mcmc_params_tables(calib_dirpath)
    targets = load_targets(app_name, region_name)

    plotter = StreamlitPlotter(targets)
    plot_type = st.sidebar.selectbox("Select plot type", list(PLOT_FUNCS.keys()))
    plot_func = PLOT_FUNCS[plot_type]

    plot_func(
        plotter,
        calib_dirpath,
        mcmc_tables,
        mcmc_params,
        targets,
        app_name,
        region_name,
    )

    path_name = os.path.join(calib_dirpath, "saved_plots")
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    with st.spinner("Saving files..."):
        file_plotter = FilePlotter(
            path_name,
            targets)
        plot_func(
            file_plotter, calib_dirpath, mcmc_tables, mcmc_params, targets, app_name, region_name
        )
