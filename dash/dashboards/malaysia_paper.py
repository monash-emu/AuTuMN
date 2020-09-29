from importlib import import_module

import streamlit as st
import pandas as pd
from autumn import db, plots
from autumn.constants import Region
from autumn.plots.plotter import StreamlitPlotter
from apps import covid_19
from typing import List
from dash import selectors
from autumn.tool_kit.params import load_targets


BASE_DATE = pd.datetime(2019, 12, 31)


PLOT_FUNCS = {}


def plot_acceptance_ratio(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
):
    plots.calibration.plots.plot_acceptance_ratio(plotter, mcmc_tables)


PLOT_FUNCS["Acceptance ratio"] = plot_acceptance_ratio


def run_dashboard():
    app_name, app_dirpath = \
        "covid_19", "C:/Users/jtrauer/AuTuMN/data/outputs/calibrate/covid_19"
    region_name = "malaysia"
    calib_name, calib_dirpath = \
        selectors.calibration_run("C:/Users/jtrauer/AuTuMN/data/outputs/calibrate/covid_19/malaysia")
    if not calib_name:
        st.write("No model run folder found to contain the data to plot here")
        return

    # Load MCMC tables
    mcmc_tables = db.load.load_mcmc_tables(calib_dirpath)
    mcmc_params = db.load.load_mcmc_params_tables(calib_dirpath)
    targets = load_targets(app_name, region_name)

    # Plot
    plotter = StreamlitPlotter(targets)
    st.sidebar.selectbox("Select plot type", list(PLOT_FUNCS.keys()))
    plot_func = plot_acceptance_ratio
    plot_func(plotter, calib_dirpath, mcmc_tables, mcmc_params, targets)


@st.cache
def get_url_df(url):
    return pd.read_csv(url)
