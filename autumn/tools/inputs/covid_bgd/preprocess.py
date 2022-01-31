from operator import index
from venv import create
import pandas as pd
from pathlib2 import Path

from autumn.tools.db import Database
from autumn.tools.utils.utils import create_date_index, COVID_BASE_DATETIME

from .fetch import COXS_VAC_DATA, COXS_DATA


def preprocess_covid_bgd(input_db: Database):

    df = process_coxs_bazar_vaccination(COXS_VAC_DATA)
    input_db.dump_df("coxs_bazar_vacc", df)
    df = process_coxs_bazar(COXS_DATA)
    input_db.dump_df("covid_coxs_bazar", df)


def process_coxs_bazar_vaccination(COXS_VAC_DATA: Path) -> pd.DataFrame:
    df = pd.read_excel(COXS_VAC_DATA, usecols=[4, 5, 7, 8], skipfooter=1)
    df.rename(
        columns={"# Total Vaccinated": "total_vaccinated", "First Dose": "dose"}, inplace=True
    )
    df = df.groupby(["Date", "dose"], as_index=False).sum()
    df = create_date_index(COVID_BASE_DATETIME, df, "Date")
    df.replace({"1st Dose": 1, "2nd Dose": 2}, inplace=True)

    return df


def process_coxs_bazar(COXS_DATA: Path) -> pd.DataFrame:

    df = pd.read_excel(COXS_DATA, skipfooter=1, usecols=[1, 2, 3, 4, 5, 6])
    df = create_date_index(COVID_BASE_DATETIME, df, "Unnamed: 1")

    return df
