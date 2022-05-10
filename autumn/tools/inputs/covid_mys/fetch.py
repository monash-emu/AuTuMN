"""
This file imports vaccination data from MOH https://github.com/MoH-Malaysia/covid19-public.
"""
import pandas as pd
from pathlib import Path
from autumn.settings import INPUT_DATA_PATH

# From MOH https://github.com/MoH-Malaysia/covid19-public
DATA_URL = (
    "https://raw.githubusercontent.com/MoH-Malaysia/covid19-public/main/vaccination/vax_demog_age_children.csv",
    "https://raw.githubusercontent.com/MoH-Malaysia/covid19-public/main/vaccination/vax_demog_age.csv",  # shareable link for sheet 07 testing data
)

COVID_MYS_DIRPATH = Path(INPUT_DATA_PATH) / "covid_mys"

COVID_MYS_VAC_CSV =  [ COVID_MYS_DIRPATH / Path(f).name for f in DATA_URL ]

def fetch_covid_mys_data():

    for file in DATA_URL:
        file_name = Path(file).name
        pd.read_csv(file).to_csv(COVID_MYS_DIRPATH / file_name, index=False)
