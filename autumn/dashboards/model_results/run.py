"""
Streamlit web UI for plotting model outputs
"""
import os

import streamlit as st

from autumn.tools import db
from autumn.tools.streamlit import selectors
from autumn.tools.plots.plotter import StreamlitPlotter

from .plots import PLOT_FUNCS


def run_dashboard():
    project = selectors.project()
    if not project:
        return

    model_run_path = selectors.model_run_path(project)
    if not model_run_path:
        msg = f"No model run outputs folder found for {project.model_name} {project.region_name}"
        st.write(msg)
        return

    # Get database from model data dir.
    db_path = os.path.join(model_run_path, "outputs.db")
    if not os.path.exists(db_path):
        db_path = os.path.join(model_run_path, "outputs")

    models = db.load.load_models_from_database(project, db_path)

    # Create plotter which will write to streamlit UI.
    plotter = StreamlitPlotter(project.plots)

    # Get user to select plot type / scenario
    plot_type = st.sidebar.selectbox("Select plot type", list(PLOT_FUNCS.keys()))
    plot_func = PLOT_FUNCS[plot_type]
    plot_func(plotter, project, models)
