"""
This file imports Google mobility data and saves it to disk as a CSV.
"""

from pathlib import Path

from autumn.settings import INPUT_DATA_PATH

INPUT_DATA_PATH = Path(INPUT_DATA_PATH)


COVID_AU_DIRPATH = INPUT_DATA_PATH / "covid_au"
NT_DATA = COVID_AU_DIRPATH / "NT-data.secret.xlsx"
NT_VAC_CSV = COVID_AU_DIRPATH / "NT-vaccination.secret.csv"


def fetch_covid_au_data():
    pass
