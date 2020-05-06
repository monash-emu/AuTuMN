"""
Streamlit web UI for plotting MCMC outputs
"""
import os
from typing import List

import pandas as pd
import streamlit as st

from autumn.db import Database
from autumn.plots import plots
from autumn.plots.plotter import StreamlitPlotter

from . import selectors, utils


def run_mcmc_plots():
    app_dirname, app_dirpath = selectors.app()
    calib_dirname, calib_dirpath = selectors.calibration_run(app_dirpath)

    # Load MCMC tables
    mcmc_tables = []
    db_paths = [
        os.path.join(calib_dirpath, f) for f in os.listdir(calib_dirpath) if f.endswith(".db")
    ]
    for db_path in db_paths:
        db = Database(db_path)
        mcmc_tables.append(db.db_query("mcmc_run"))

    plotter = StreamlitPlotter({})
    plot_type = st.sidebar.selectbox("Select plot type", list(PLOT_FUNCS.keys()))
    plot_func = PLOT_FUNCS[plot_type]
    plot_func(plotter, mcmc_tables)


def plot_posterior(plotter: StreamlitPlotter, mcmc_tables: List[pd.DataFrame]):
    non_param_cols = ["idx", "Scenario", "loglikelihood", "accept"]
    param_options = [c for c in mcmc_tables[0].columns if c not in non_param_cols]
    num_bins = st.sidebar.slider("Number of bins", 1, 50, 10)
    chosen_param = st.sidebar.selectbox("Select parameter", param_options)
    plots.plot_posterior(plotter, mcmc_tables, chosen_param, num_bins)


PLOT_FUNCS = {
    "Posterior distributions": plot_posterior,
}
