from numpy.lib.shape_base import column_stack
import pandas as pd

from autumn.tools.db import Database

from .fetch import COVID_LKA_CSV

COVID_BASE_DATETIME = pd.datetime(2019, 12, 31)


def preprocess_covid_lka(input_db: Database):

    df = pd.read_csv(COVID_LKA_CSV)
    df.periodname = pd.to_datetime(
        df.periodname, errors="coerce", format="%Y-%m-%d", infer_datetime_format=False
    )
    df["date_index"] = (df.periodname - COVID_BASE_DATETIME).dt.days
    df = df[df.periodname <= pd.to_datetime("today")]
    df.rename(columns={"PCR tests done": "PCR_tests_done"}, inplace=True)

    input_db.dump_df("covid_lka", df)
