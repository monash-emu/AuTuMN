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
import os

# Plus fix Streamlit hot reloading, which requires PYTHONPATH hacks
# https://github.com/streamlit/streamlit/issues/1176
MODULE_DIRNAMES = ["summer", "autumn"]
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
dirpaths = []
for module_dirname in MODULE_DIRNAMES:
    dirpaths += [
        os.path.join(PROJECT_PATH, d)
        for d, _, _ in os.walk(module_dirname)
        if not "__pycache__" in d
    ]

pypath = os.environ.get("PYTHONPATH")
if pypath:
    dirpaths = pypath.split(":") + dirpaths

os.environ["PYTHONPATH"] = ":".join(dirpaths)

from autumn.outputs.streamlit_plots import main

main()
