import pandas as pd
from autumn.core.db import Database
from autumn.core.utils.utils import create_date_index
from autumn.settings.constants import COVID_BASE_DATETIME

from .fetch import NT_DATA, NT_VAC_CSV


def preprocess_covid_au(input_db: Database):
    df = pd.read_excel(NT_DATA, sheet_name="Testing", skiprows=[0], skipfooter=2, usecols=[1, 4])
    df = create_date_index(COVID_BASE_DATETIME, df, "testdate")
    input_db.dump_df("covid_nt", df)
    df = pd.read_csv(NT_VAC_CSV)
    df = create_date_index(COVID_BASE_DATETIME, df, "date")
    input_db.dump_df("covid_nt_vac.secret", df)
