"""
Streamlit web UI for plotting model outputs
"""
import os
from importlib import import_module

import streamlit as st

from autumn.tools.plots.plotter import StreamlitPlotter
from dash import selectors

from .plots import PLOT_FUNCS


def run_dashboard():
    app_name, _ = selectors.app_name(run_type="model")
    if not app_name:
        st.write("No application folder found")
        return

    region_name, _ = selectors.app_region_name(app_name)
    if not region_name:
        st.write("No parameter set folder found")
        return

    # Import the app so we can re-build the model if we need to
    app_module = import_module(f"apps.{app_name}")
    app_region = app_module.app.get_region(region_name)

    # Create plotter which will write to streamlit UI.
    plotter = StreamlitPlotter(app_region.targets)

    # Get user to select plot type / scenario
    plot_type = st.sidebar.selectbox("Select plot type", list(PLOT_FUNCS.keys()))
    plot_func = PLOT_FUNCS[plot_type]
    plot_func(plotter, app_region)
