"""
This file imports fetches kiribati data and saves it to disk as a CSV.
"""

from pathlib import Path

from autumn.settings import INPUT_DATA_PATH

INPUT_DATA_PATH = Path(INPUT_DATA_PATH)
KIR_BCG_COV = INPUT_DATA_PATH / "tb_kir" / "bcg.csv"


def fetch_covid_bgd_data():
    pass
