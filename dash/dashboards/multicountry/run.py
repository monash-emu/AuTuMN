"""
Streamlit web UI for plotting model outputs
"""

import streamlit as st
import os

from autumn import db
from autumn.tool_kit.params import load_targets
from autumn.plots.plotter import StreamlitPlotter

from dash import selectors
from .plots import PLOT_FUNCS


def run_dashboard():
    app_name, app_dirpath = selectors.app_name(run_type="calibrate")
    if not app_name:
        st.write("No calibrations have been run yet")
        return

    region_names = {}
    region_dirpaths = {}

    for i_region in range(1, 3):
        region_names[i_region], region_dirpaths[i_region] = \
            selectors.output_region_name(app_dirpath, f"Select region #{str(i_region)}")
        if not region_names[i_region]:
            st.write("No region folder found")
            return

    calib_name, calib_dirpath = selectors.calibration_run(region_dirpaths[1])
    if not calib_name:
        st.write("No model run folder found")
        return

    # Load MCMC tables
    mcmc_tables = db.load.load_mcmc_tables(calib_dirpath)
    mcmc_params = db.load.load_mcmc_params_tables(calib_dirpath)
    targets = load_targets(app_name, region_names[1])

    plotter = StreamlitPlotter(targets)
    plot_type = st.sidebar.selectbox("Select plot type", list(PLOT_FUNCS.keys()))
    plot_func = PLOT_FUNCS[plot_type]
    plot_func(plotter, calib_dirpath, mcmc_tables, mcmc_params, targets, app_name, region_names[1])
