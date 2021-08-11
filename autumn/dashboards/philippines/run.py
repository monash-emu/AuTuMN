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
    project = get_project(Models.COVID_19, Region.MANILA)
    if not project:
        return

    calib_path = selectors.calibration_path(project)

    if not calib_path:
        st.write("No calibration run folder found")
        return

    # Load MCMC tables
    mcmc_tables = db.load.load_mcmc_tables(calib_path)
    mcmc_params = db.load.load_mcmc_params_tables(calib_path)

    # Plot in streamlit.
    plotter = StreamlitPlotter(project.plots)
    dash.select_plot(
        plotter,
        calib_path,
        mcmc_tables,
        mcmc_params,
        project.plots,
        project.model_name,
        project.region_name,
    )
