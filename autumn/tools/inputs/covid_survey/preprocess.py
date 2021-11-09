import pandas as pd
import os

from autumn.tools.db import Database
from autumn.tools.utils.utils import create_date_index, COVID_BASE_DATETIME

from autumn.settings import INPUT_DATA_PATH

from .fetch import COVID_SURVEY_PATH

CSV_FILES = [
    os.path.join(INPUT_DATA_PATH, "covid_survey", file) for file in os.listdir(COVID_SURVEY_PATH)
]
MASKS = [file for file in CSV_FILES if "mask_" in file]


def preprocess_covid_survey(input_db: Database):
    df = get_mask()
    input_db.dump_df("survey_mask", df)

def get_mask():
    df = pd.concat(map(pd.read_csv, MASKS))
    df.loc[:, "survey_date"] = pd.to_datetime(
        df["survey_date"], format="%Y%m%d", infer_datetime_format=False
    )
    create_date_index(COVID_BASE_DATETIME, df, "survey_date")
    df = df[
        [
            "iso_code",
            "region",
            "date",
            "date_index",
            "sample_size",
            "percent_mc",
            "mc_se",
            "percent_mc_unw",
            "mc_se_unw",
        ]
    ]

    return df
