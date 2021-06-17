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

from autumn.dashboards.calibration_results.run import run_dashboard as calibration_results
from autumn.dashboards.model_internals.run import run_dashboard as model_internals
from autumn.dashboards.model_results.run import run_dashboard as model_results

# from autumn.dashboards.multicountry_inputs import (
#     run_dashboard as run_multicountry_inputs,
# )

# from autumn.dashboards.multicountry_manual import (
#     run_dashboard as run_multicountry_manual,
# )
# from autumn.dashboards.multicountry_plots import (
#     run_dashboard as run_multicountry_dashboard,
# )

# from autumn.dashboards.multicountry_uncertainty import (
#     run_dashboard as run_multicountry_uncertainty,
# )
from autumn.dashboards.philippines import run_dashboard as run_philippines

from autumn.dashboards.vic_second_wave_paper.run import run_dashboard as run_vic_paper

DASHBOARDS = {
    "Home": None,
    "Calibration results": calibration_results,
    "Model internals": model_internals,
    "Model results": model_results,
    # "Multi-country manual": run_multicountry_manual,
    # "Multi-country plots": run_multicountry_dashboard,
    # "Multi-country inputs": run_multicountry_inputs,
    # "Multi-country uncertainty": run_multicountry_uncertainty,
    "Vic 2nd wave paper": run_vic_paper,
    "Philippines COVID": run_philippines,
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
    - **Calibration results**: Inspect the outputs of a model calibration
    - **Model internals**: Inspect the values inside a model before it is run
    - **Model results**: Inspect the model outputs after it has been run
    - **Philippines**: Inspect results for Philippines covid application
    """
        # - **Multi-country**: Multi-country plots
        # - **Vic 2nd wave paper**: Inspect results for Victoria application
    )
