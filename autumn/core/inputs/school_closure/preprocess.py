from pathlib import Path

import pandas as pd
from autumn.core.db import Database
from autumn.core.utils.utils import create_date_index
from autumn.settings import COVID_BASE_DATETIME

from .fetch import UNESCO_SCHOOL_CLOSURE_CSV


def preprocess_school_closure(input_db: Database):

    df = process_school_closure(UNESCO_SCHOOL_CLOSURE_CSV)
    input_db.dump_df("school_closure", df)


def process_school_closure(data_path: Path) -> pd.DataFrame:
    """Processes school closure data
    and dumps it into the parquet db.
    """
    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df.rename(columns=lambda x: x.lower().replace(" ", "_"), inplace=True)
    df = create_date_index(COVID_BASE_DATETIME,df,'date')

    return df

