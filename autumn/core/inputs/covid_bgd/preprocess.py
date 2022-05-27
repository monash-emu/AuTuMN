from typing import List
import pandas as pd


from autumn.core.db import Database
from autumn.core.utils.utils import create_date_index
from autumn.settings.constants import COVID_BASE_DATETIME

from .fetch import COXS_VAC_DATA, COXS_DATA, VACC_FILE


def preprocess_covid_bgd(input_db: Database):

    df = process_coxs_bazar_vaccination(COXS_VAC_DATA)
    input_db.dump_df("coxs_bazar_vacc", df)
    df = process_coxs_bazar(COXS_DATA)
    input_db.dump_df("covid_coxs_bazar", df)
    df = process_bgd_dhk_vaccination(VACC_FILE)
    input_db.dump_df("bgd_vacc", df)


def process_coxs_bazar_vaccination(COXS_VAC_DATA: str) -> pd.DataFrame:
    df = pd.read_excel(COXS_VAC_DATA, usecols=[4, 5, 7, 8], skipfooter=1)
    df.rename(
        columns={"# Total Vaccinated": "total_vaccinated", "First Dose": "dose"},
        inplace=True,
    )
    df = df.groupby(["Date", "dose"], as_index=False).sum()
    df = create_date_index(COVID_BASE_DATETIME, df, "Date")
    df.replace({"1st Dose": 1, "2nd Dose": 2}, inplace=True)

    return df


def process_coxs_bazar(COXS_DATA: str) -> pd.DataFrame:

    df = pd.read_excel(COXS_DATA, skipfooter=1, usecols=[1, 2, 3, 4, 5, 6])
    df = create_date_index(COVID_BASE_DATETIME, df, "Unnamed: 1")

    return df


def process_bgd_dhk_vaccination(VACC_FILE: List) -> pd.DataFrame:

    df = pd.DataFrame()

    for file in VACC_FILE:
        file_info = file.stem.split("\\")[-1].split("_")
        dose = file_info[1]
        region = file_info[2]

        tmp_df = pd.read_csv(file)
        cols = list(tmp_df.columns)
        cols.remove("Category")

        tmp_df["total"] = tmp_df.loc[:, cols].sum(axis=1)
        tmp_df[["total"] + cols] = tmp_df.loc[:, ["total"] + cols].cumsum()

        tmp_df["dose"] = dose
        tmp_df["region"] = region

        df = df.append(tmp_df)

    df = create_date_index(COVID_BASE_DATETIME, df, "Category")
    return df
