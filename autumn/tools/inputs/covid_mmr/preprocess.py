import pandas as pd

from autumn.tools.db import Database
from .fetch import COVID_MMR_TESTING_CSV
from autumn.tools.utils.utils import create_date_index

from autumn.settings.constants import COVID_BASE_DATETIME

def preprocess_covid_mmr(input_db: Database):

    df = get_mmr_data(COVID_MMR_TESTING_CSV)
    df = df[['date','date_index','tests']]
    input_db.dump_df("covid_mmr", df)

def get_mmr_data(path_to_csv):

    df = pd.read_csv(COVID_MMR_TESTING_CSV)
   
    str_col = ["Tests", "Cases", "Recovered", "Negative"]

    df[str_col] = df[str_col].replace(to_replace=r",", value="", regex=True)
    df[str_col] = df[str_col].apply(pd.to_numeric)
    df["Date"] = pd.to_datetime(df["Date"])
    df = create_date_index(COVID_BASE_DATETIME, df, "Date")

    return df


