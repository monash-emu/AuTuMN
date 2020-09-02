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


Website: https://www.streamlit.io/
Docs: https://docs.streamlit.io/
"""
import sys

from dash.dashboards import calibration
from dash.dashboards import scenario
from dash.dashboards import model


if len(sys.argv) > 1 and sys.argv[1] == "mcmc":
    calibration.run_dashboard()
elif len(sys.argv) > 1 and sys.argv[1] == "scenario":
    scenario.run_dashboard()
else:
    model.run_dashboard()
