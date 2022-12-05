import pandas as pd
from autumn.core.db import Database

from .fetch import SERO_SURVEY_CSV


def preprocess_covid_serosurvey(input_db: Database):

    df = pd.read_csv(SERO_SURVEY_CSV)
    input_db.dump_df("sero-survey", df)
