import pandas as pd
import os
import numpy as np

from settings import INPUT_DATA_PATH
from autumn.db import Database

OUR_WORLD_IN_DATA_DIRPATH = os.path.join(INPUT_DATA_PATH, "owid")
OUR_WORLD_IN_DATA_CSV_PATH = os.path.join(OUR_WORLD_IN_DATA_DIRPATH, "owid-covid-data.csv")


def preprocess_our_world_in_data(input_db: Database):
    df = pd.read_csv(OUR_WORLD_IN_DATA_CSV_PATH)

    # Replace the one strange value for test numbers in Malaysia
    df.loc[(df.iso_code == "MYS") & (df.new_tests > 1e5), "new_tests"] = np.nan
    input_db.dump_df("owid", df)
