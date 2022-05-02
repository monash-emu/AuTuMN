from numpy.lib.shape_base import column_stack
import pandas as pd

from autumn.tools.db import Database

from autumn.settings.constants import COVID_BASE_DATETIME

from .fetch import COVID_LKA_CSV, COVID_LKA_2021_CSV

def preprocess_covid_lka(input_db: Database):

    df = pd.read_csv(COVID_LKA_CSV)
    df_2021 = pd.read_csv(COVID_LKA_2021_CSV)
    df = df.append(df_2021)

    df.periodname = pd.to_datetime(
        df.periodname, errors="coerce", format="%Y-%m-%d", infer_datetime_format=False
    )
    df["date_index"] = (df.periodname - COVID_BASE_DATETIME).dt.days
    df = df[df.periodname <= pd.to_datetime("today")]
    df.rename(
        columns=lambda x: x.strip().replace("(", "").replace(")", "").replace(" ", "_"),
        inplace=True,
    )
    # df.rename(columns={"Sri Lanka PCR tests done": "Sri_Lanka_PCR_tests_done", "Western PDHS PCR tests done":"Western_PDHS_PCR_tests_done"}, inplace=True)

    input_db.dump_df("covid_lka", df)
