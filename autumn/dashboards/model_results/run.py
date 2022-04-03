"""
Streamlit web UI for plotting model outputs
"""
import os

import streamlit as st

from ... import db
from ...outputs.streamlit import selectors
from autumn.outputs.plots import StreamlitPlotter

from .plots import dash


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
    dash.select_plot(plotter, project, models)
