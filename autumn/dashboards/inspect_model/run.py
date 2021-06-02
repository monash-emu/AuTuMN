"""
Streamlit web UI for plotting model outputs
"""
import streamlit as st

from autumn.tools.plots.plotter import StreamlitPlotter
from autumn.tools.streamlit import selectors

from .plots import PLOT_FUNCS


def run_dashboard():
    project = selectors.project()
    if not project:
        return

    # Create plotter which will write to streamlit UI.
    plotter = StreamlitPlotter(project.plots)

    # Get user to select plot type / scenario
    plot_type = st.sidebar.selectbox("Select plot type", list(PLOT_FUNCS.keys()))
    plot_func = PLOT_FUNCS[plot_type]
    plot_func(plotter, project)
