import pandas as pd

from autumn.db import Database

from .fetch import COVID_AU_CSV_PATH


def preprocess_covid_au(input_db: Database):
    df = pd.read_csv(COVID_AU_CSV_PATH)
    input_db.dump_df("covid_au", df)
