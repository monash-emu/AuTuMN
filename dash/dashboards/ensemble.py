from importlib import import_module

import streamlit as st
import pandas as pd
from autumn import db, plots
from autumn.constants import Region
from autumn.plots.plotter import StreamlitPlotter
from apps import covid_19


BASE_DATE = pd.datetime(2019, 12, 31)


def run_dashboard():
    st.header("Visualise Commonwealth Victorian ensemble data")
    url = st.text_input("Enter the URL of the VIC ensemble CSV", value="", type="default")
    if not url.endswith(".csv"):
        st.warning(f"Not a CSV file: {url}")
        return

    df = get_url_df(url)

    plotter = StreamlitPlotter({})
    fig, axis, _, _, _ = plotter.get_figure()
    runs = df.run.unique().tolist()
    is_logscale = st.checkbox("Log scale")

    for run in runs:
        run_df = df[df["run"] == run]
        axis.plot(run_df["times"], run_df["notifications_at_sympt_onset"], alpha=0.7)

    if is_logscale:
        axis.set_yscale("log")

    plotter.save_figure(
        fig, filename="ensemble", title_text="Notifications at symptom onset - 6 week projection"
    )


@st.cache
def get_url_df(url):
    return pd.read_csv(url)
