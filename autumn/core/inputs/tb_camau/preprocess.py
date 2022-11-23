import pandas as pd

from autumn.core.db import Database

from .fetch import TB_CAMAU_CSV_PATH


def preprocess_tb_camau(input_db: Database):

    df = pd.read_csv(TB_CAMAU_CSV_PATH)
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    input_db.dump_df("tb_camau", df)
