import pandas as pd
from autumn.core.db import Database

from .fetch import GISAID_VOC


def preprocess_covid_gisaid(input_db: Database):

    df = pd.read_excel(GISAID_VOC)
    input_db.dump_df("gisaid", df)
