import pandas as pd
from autumn.core.db import Database

from .fetch import SERO_SURVEY


def preprocess_covid_serosurvey(input_db: Database):

    df = pd.read_excel(SERO_SURVEY)
    input_db.dump_df("sero-survey", df)