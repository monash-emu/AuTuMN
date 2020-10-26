"""
Application entrypoint for Streamlit plots web UI

First don't forget to activate your conda / virtual env

    Conda: conda activate <envname>
    Virtualenv Windows: ./env/Scripts/activate.ps1
    Virtualenv Linux: . env/bin/activate

Then, make sure you have streamlit installed

    pip show streamlit

If not, install project requirements

    pip install -r requirements.txt

Run from a command line shell (with env active) using

    streamlit run plots.py

Website: https://www.streamlit.io/
Docs: https://docs.streamlit.io/
"""
import streamlit as st

from dash.dashboards.model.run import run_dashboard as run_model_dashboard
from dash.dashboards.calibration.run import run_dashboard as run_calibration_dashboard
from dash.dashboards.scenario.run import run_dashboard as run_scenario_dashboard
from dash.dashboards.dhhs import run_dashboard as run_dhhs_dashboard
from dash.dashboards.ensemble import run_dashboard as run_ensemble_dashboard
from dash.dashboards.multicountry import run_dashboard as run_multicountry_dashboard

DASHBOARDS = {
    "Home": None,
    "Model internals": run_model_dashboard,
    "Calibration results": run_calibration_dashboard,
    "Model results": run_scenario_dashboard,
    "Multi-country": run_multicountry_dashboard,
    "DHHS results": run_dhhs_dashboard,
    "Ensemble results": run_ensemble_dashboard,
}


dashboards_options = list(DASHBOARDS.keys())
dashboard_key = st.sidebar.selectbox("Select a dashboard", dashboards_options)
dashboard_func = DASHBOARDS[dashboard_key]
if dashboard_func:
    dashboard_func()
else:
    st.title("Autumn dashboards")
    st.write("Select a dashboard from the sidebar. Your options are:")
    st.markdown(
        """
    - **Model internals**: Inspect the values inside a model
    - **Calibration results**: Inspect the results of a calibration
    - **Model results**: Inspect the model outputs
    - **Multi-country**: Multi-country plots
    - **DHHS results**: Inspect results which will be sent to DHHS
    - **Ensemble results**: Inspect results which will be sent to ensemble forecasting
    """
    )
