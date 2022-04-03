"""
Streamlit web UI for plotting MCMC outputs
"""
import streamlit as st

from ... import db
from autumn.outputs.plots import StreamlitPlotter
from ...outputs.streamlit import selectors

from .plots import dash


def run_dashboard():
    project = selectors.project()
    if not project:
        return

    calib_path = selectors.calibration_path(project)
    if not calib_path:
        msg = f"No calibration outputs folder found for {project.model_name} {project.region_name}"
        st.write(msg)
        return

    # Load MCMC tables
    mcmc_tables = db.load.load_mcmc_tables(calib_path)
    mcmc_params = db.load.load_mcmc_params_tables(calib_path)

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
