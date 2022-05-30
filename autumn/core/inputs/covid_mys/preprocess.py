import pandas as pd

from autumn.core.db import Database

from .fetch import COVID_MYS_DIRPATH, COVID_MYS_VAC_CSV

from autumn.settings.constants import COVID_BASE_DATETIME
from autumn.core.utils.utils import create_date_index

FILTER_COL = {
    "partial_5_11",
    "full_5_11",
    "booster_5_11",
    "partial_12_17",
    "full_12_17",
    "booster_12_17",
    "partial_missing",
    "full_missing",
    "booster_missing",
}


def preprocess_covid_mys(input_db: Database):

    df = combine_vac_df()
    input_db.dump_df("covid_mys_vac", df)


def get_vac_df(file):
    df = pd.read_csv(file)
    df = df.drop(columns=[col for col in df.columns if col in FILTER_COL])
    df = df.groupby(["date"], as_index=False).sum()
    df = create_date_index(COVID_BASE_DATETIME, df=df, datecol="date")
    return df


def combine_vac_df():
    df1 = get_vac_df(COVID_MYS_VAC_CSV[0])
    df2 = get_vac_df(COVID_MYS_VAC_CSV[1])

    df = pd.merge(df2, df1, how="left", on=["date_index", "date"])
    df = df.melt(["date", "date_index"], value_name="number")

    df["dose"] = df["variable"].apply(lambda s: s.split("_")[0])
    df["start_age"] = df["variable"].apply(lambda s: int(s.split("_")[1]))
    df["end_age"] = df["variable"].apply(
        lambda s: int(x[2]) if (len(x := s.split("_")) == 3) else int(x[1]) + 1
    )
    df = df.drop(columns="variable")
    return df
