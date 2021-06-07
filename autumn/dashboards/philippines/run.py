"""
Streamlit web UI for plotting MCMC outputs
"""
import os

import streamlit as st

from autumn.tools import db
from autumn.tools.plots.plotter import FilePlotter, StreamlitPlotter
from autumn.tools.streamlit import selectors
from autumn.tools.project import get_project
from autumn.settings import Models, Region

from .plots import dash


def run_dashboard():
    project = get_project(Models.COVID_19, Region.PHILIPPINES)
    if not project:
        return

    calib_dirpath = selectors.calibration_path(project)
    if not calib_dirpath:
        st.write("No calibration run folder found")
        return

    # Load MCMC tables
    mcmc_tables = db.load.load_mcmc_tables(calib_dirpath)
    mcmc_params = db.load.load_mcmc_params_tables(calib_dirpath)

    # Plot in streamlit.
    plotter = StreamlitPlotter(project.plots)
    plot_name = dash.select_plot(plotter, calib_dirpath, mcmc_tables, mcmc_params, project)

    # Plot to filesystem
    path_name = os.path.join(calib_dirpath, "saved_plots")
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    with st.spinner("Saving files..."):
        file_plotter = FilePlotter(path_name, project.plots)
        dash.plot_funcs[plot_name](file_plotter, calib_dirpath, mcmc_tables, mcmc_params, project)
