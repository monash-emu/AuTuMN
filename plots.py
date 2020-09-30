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
    streamlit run plots.py mcmc
    streamlit run plots.py scenario
    streamlit run plots.py dhhs
    streamlit run plots.py ensemble


Website: https://www.streamlit.io/
Docs: https://docs.streamlit.io/
"""

import sys

from dash.dashboards.model.run import run_dashboard as run_model_dashboard
from dash.dashboards.calibration.run import run_dashboard as run_calibration_dashboard
from dash.dashboards.scenario.run import run_dashboard as run_scenario_dashboard
from dash.dashboards.dhhs import run_dashboard as run_dhhs_dashboard
from dash.dashboards.ensemble import run_dashboard as run_ensemble_dashboard
from dash.dashboards.malaysia_paper import run_dashboard as run_malaysia_dashboard

if len(sys.argv) > 1 and sys.argv[1] == "mcmc":
    run_calibration_dashboard()
elif len(sys.argv) > 1 and sys.argv[1] == "scenario":
    run_scenario_dashboard()
elif len(sys.argv) > 1 and sys.argv[1] == "dhhs":
    run_dhhs_dashboard()
elif len(sys.argv) > 1 and sys.argv[1] == "ensemble":
    run_ensemble_dashboard()
elif len(sys.argv) > 1 and sys.argv[1] == "malaysia_paper":
    run_malaysia_dashboard()
else:
    run_model_dashboard()
