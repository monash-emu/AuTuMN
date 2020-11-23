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


from dash.dashboards.inspect_model.run import run_dashboard as inspect_model_dashboard
from dash.dashboards.calibration_results.run import run_dashboard as calibration_results_dashboard
from dash.dashboards.model_results.run import run_dashboard as run_scenario_dashboard
from dash.dashboards.dhhs import run_dashboard as run_dhhs_dashboard
from dash.dashboards.ensemble import run_dashboard as run_ensemble_dashboard
from dash.dashboards.multicountry_plots import run_dashboard as run_multicountry_dashboard
from dash.dashboards.multicountry_inputs import run_dashboard as run_multicountry_inputs
from dash.dashboards.run_model import run_dashboard as run_model_dashboard
from dash.dashboards.run_calibrate import run_dashboard as run_calibrate_dashboard
from dash.dashboards.multicountry_uncertainty import run_dashboard as run_multicountry_uncertainty


DASHBOARDS = {
    "Home": None,
    "Model internals": inspect_model_dashboard,
    "Run a model": run_model_dashboard,
    "Model results": run_scenario_dashboard,
    "Calibrate a model": run_calibrate_dashboard,
    "Calibration results": calibration_results_dashboard,
    "Multi-country plots": run_multicountry_dashboard,
    "Multi-country inputs": run_multicountry_inputs,
    "Multi-country uncertainty": run_multicountry_uncertainty,
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
    - **Model internals**: Inspect the values inside a model before it is run
    - **Run**: Run a model
    - **Model results**: Inspect the model outputs after it has been run
    - **Calibrate**: Calibrate a model
    - **Calibration results**: Inspect the outputs of a model calibration
    - **Multi-country**: Multi-country plots
    - **DHHS results**: Inspect results which will be sent to DHHS
    - **Ensemble results**: Inspect results which will be sent to ensemble forecasting
    """
    )
