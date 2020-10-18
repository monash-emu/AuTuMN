import streamlit as st

PLOT_FUNCS = {}


def test_only(calib_dir_path, mcmc_tables, targets):
    st.write("This is only a test at this stage")


PLOT_FUNCS["Test only"] = test_only
