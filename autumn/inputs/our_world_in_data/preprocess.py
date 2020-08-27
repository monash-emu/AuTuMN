import pandas as pd
import os

from autumn import constants
from autumn.db import Database

OUR_WORLD_IN_DATA_DIRPATH = os.path.join(constants.INPUT_DATA_PATH, "our-world-in-data")
OUR_WORLD_IN_DATA_CSV_PATH = os.path.join(OUR_WORLD_IN_DATA_DIRPATH, "owid-covid-data.csv")


def preprocess_our_world_in_data(input_db: Database):
    df = pd.read_csv(OUR_WORLD_IN_DATA_CSV_PATH)
    input_db.dump_df("our-world-in-data", df)
