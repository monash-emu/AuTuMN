"""
Streamlit web UI for plotting model outputs
"""

from autumn.outputs.plots import StreamlitPlotter
from ...outputs.streamlit import selectors

from .plots import dash


def run_dashboard():
    project = selectors.project()
    if not project:
        return

    # Create plotter which will write to streamlit UI.
    plotter = StreamlitPlotter(project.plots)

    # Get user to select plot type / scenario
    dash.select_plot(plotter, project)
