"""
This file download the latest data for Myanmar
"""

import os
import pandas as pd
from autumn.settings import INPUT_DATA_PATH

COVID_MMR_TESTING_CSV = os.path.join(INPUT_DATA_PATH, "covid_mmr", "cases.csv")

URL = "https://docs.google.com/spreadsheets/d/1VeUof9_-s0bsndo8tLsCwnAhkUUZgsdV-r980gumMPA/export?format=csv&id=1VeUof9_-s0bsndo8tLsCwnAhkUUZgsdV-r980gumMPA"


def fetch_covid_mmr_data():

    mmr_df = pd.read_csv(URL)
    mmr_df.to_csv(COVID_MMR_TESTING_CSV)
