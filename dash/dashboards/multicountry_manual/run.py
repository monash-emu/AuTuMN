"""
Streamlit web UI for plotting model outputs
"""

import streamlit as st
import os

from autumn import db
from autumn.plots.plotter import StreamlitPlotter
from autumn.utils.params import load_targets
from dash import selectors
from importlib import import_module

from .plots import PLOT_FUNCS


def run_dashboard():
    app_name, app_dirpath = selectors.app_name(run_type="run")
    if not app_name:
        st.write("No applications have been run yet")
        return

    # Prelims
    n_countries = st.sidebar.slider("Number of countries", 2, 6, 6)
    (
        region_names,
        region_dirpaths,
        run_datestrs,
        run_dirpaths,
        scenarios,
        targets,
    ) = ({}, {}, {}, {}, {}, {})

    for i_region in range(n_countries):

        # Get regions for comparison
        region_names[i_region], region_dirpaths[i_region] = selectors.output_region_name(
            app_dirpath, f"Select region #{str(i_region)}", i_region
        )
        if not region_names[i_region]:
            st.write("No region folder found")
            return

        # Specific run name and path
        run_datestrs[i_region], run_dirpaths[i_region] = selectors.model_run(region_dirpaths[i_region], i_region)
        if not run_datestrs[i_region]:
            st.write("No model run folder found")
            return

        # Get database from model data dir.
        db_path = os.path.join(run_dirpaths[i_region], "outputs.db")
        if not os.path.exists(db_path):
            db_path = os.path.join(run_dirpaths[i_region], "outputs")

        app_module = import_module(f"apps.{app_name}")
        app_region = app_module.app.get_region(region_names[i_region])

        scenarios[i_region] = db.load.load_model_scenarios(db_path, app_region.params)

        targets[i_region] = load_targets(app_name, region_names[i_region])

        plot_type = "Multi-country manual"
        plot_func = PLOT_FUNCS[plot_type]

    plotter = StreamlitPlotter(targets[0])
    plot_func(plotter, scenarios, targets, app_name, region_names)
