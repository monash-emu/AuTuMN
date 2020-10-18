import streamlit as st

PLOT_FUNCS = {}


def test_only(plotter, calib_dir_path, mcmc_tables, mcmc_params, targets, app_name, region_name):
    st.write("This is only a test at this stage")


PLOT_FUNCS["Test only"] = test_only
