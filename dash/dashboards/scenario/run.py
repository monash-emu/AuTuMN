"""
Streamlit web UI for plotting model outputs
"""
import os
from importlib import import_module

import streamlit as st

from autumn.db.models import load_model_scenarios
from autumn.plots.plotter import StreamlitPlotter

from dash import selectors
from .plots import PLOT_FUNCS


def run_dashboard():
    app_name, app_dirpath = selectors.app_name(run_type="run")
    if not app_name:
        st.write("No applications have been run yet")
        return

    region_name, region_dirpath = selectors.output_region_name(app_dirpath)
    if not region_name:
        st.write("No parameter set folder found")
        return

    run_datestr, run_dirpath = selectors.model_run(region_dirpath)
    if not run_datestr:
        st.write("No model run folder found")
        return

    # Import the app so we can re-build the model if we need to
    app_module = import_module(f"apps.{app_name}")
    app_region = app_module.app.get_region(region_name)

    # Get database from model data dir.
    db_path = os.path.join(run_dirpath, "outputs.db")
    scenarios = load_model_scenarios(db_path, app_region.params)

    # Create plotter which will write to streamlit UI.
    plotter = StreamlitPlotter(app_region.targets)

    # Get user to select plot type / scenario
    plot_type = st.sidebar.selectbox("Select plot type", list(PLOT_FUNCS.keys()))
    plot_func = PLOT_FUNCS[plot_type]
    plot_func(plotter, app_region, scenarios)
