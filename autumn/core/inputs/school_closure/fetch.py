"""
This file imports fetches kiribati data and saves it to disk as a CSV.
"""
from pathlib import Path

import pandas as pd
from autumn.settings import INPUT_DATA_PATH

INPUT_DATA_PATH = Path(INPUT_DATA_PATH)
UNESCO_SCHOOL_CLOSURE_XLS = (
    INPUT_DATA_PATH / "school-closure" / "UNESCO_school_closures_database.xlsx"
)
UNESCO_SCHOOL_CLOSURE_CSV = INPUT_DATA_PATH / "school-closure" / "school_closure.csv"


def fetch_school_closure_data() -> None:
    pd.read_excel(UNESCO_SCHOOL_CLOSURE_XLS).to_csv(
        UNESCO_SCHOOL_CLOSURE_CSV, index=False
    )
    return None
