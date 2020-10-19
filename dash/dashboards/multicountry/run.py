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

    # Prelims
    n_countries = st.sidebar.slider("Number of countries", 1, 5, 2)
    region_names, region_dirpaths, calib_names, calib_dirpaths, mcmc_tables, mcmc_params, targets = \
        {}, {}, {}, {}, {}, {}, {}

    for i_region in range(1, n_countries + 1):

        # Get regions for comparison
        region_names[i_region], region_dirpaths[i_region] = \
            selectors.output_region_name(app_dirpath, f"Select region #{str(i_region)}")
        if not region_names[i_region]:
            st.write("No region folder found")
            return

        # Specific calibration run name and path
        calib_names[i_region], calib_dirpaths[i_region] = \
            selectors.calibration_run(region_dirpaths[i_region], f"Select region #{str(i_region)}")
        if not calib_names[i_region]:
            st.write("No model run folder found")
            return

        # Load MCMC tables
        mcmc_tables[i_region] = db.load.load_mcmc_tables(calib_dirpaths[i_region])
        mcmc_params[i_region] = db.load.load_mcmc_params_tables(calib_dirpaths[i_region])
        targets[i_region] = load_targets(app_name, region_names[i_region])

    plotter = StreamlitPlotter(targets[1])
    plot_type = st.sidebar.selectbox("Select plot type", list(PLOT_FUNCS.keys()))
    plot_func = PLOT_FUNCS[plot_type]
    plot_func(plotter, calib_dirpaths[1], mcmc_tables, mcmc_params, targets, app_name, region_names[1])
