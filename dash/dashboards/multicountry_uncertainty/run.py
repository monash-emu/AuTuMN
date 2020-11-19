"""
Streamlit web UI for plotting model outputs
"""

import streamlit as st

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
    n_countries = st.sidebar.slider("Number of countries", 2, 6, 3)
    (
        region_names,
        region_dirpaths,
        calib_names,
        calib_dirpaths,
        mcmc_tables,
        mcmc_params,
        targets,
    ) = ({}, {}, {}, {}, {}, {}, {})

    for i_region in range(n_countries):

        # Get regions for comparison
        region_names[i_region], region_dirpaths[i_region] = selectors.output_region_name(
            app_dirpath, f"Select region #{str(i_region)}"
        )
        if not region_names[i_region]:
            st.write("No region folder found")
            return

        # Specific calibration run name and path
        calib_names[i_region], calib_dirpaths[i_region] = selectors.calibration_run(
            region_dirpaths[i_region], f"Select region #{str(i_region)}"
        )
        if not calib_names[i_region]:
            st.write("No model run folder found")
            return

        # Load MCMC tables
        mcmc_tables[i_region] = db.load.load_mcmc_tables(calib_dirpaths[i_region])
        mcmc_params[i_region] = db.load.load_mcmc_params_tables(calib_dirpaths[i_region])
        targets[i_region] = load_targets(app_name, region_names[i_region])

        plot_type = "Multi-country uncertainty"
        plot_func = PLOT_FUNCS[plot_type]

    plotter = StreamlitPlotter(targets[0])
    plot_func(plotter, calib_dirpaths, mcmc_tables, mcmc_params, targets, app_name, region_names)
