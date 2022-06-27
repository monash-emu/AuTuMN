from pathlib import Path

import pandas as pd
from autumn.core.db import Database

from .fetch import KIR_BCG_COV


def preprocess_tb_kir(input_db: Database):

    df = process_kiribati_bcg(KIR_BCG_COV)
    input_db.dump_df("tb_kir_bcg_cov", df)


def process_kiribati_bcg(data_path: Path) -> pd.DataFrame:
    """Processes Kiribati BCG vaccination coverage data
    and dumps it into the parquet db.
    """
    df = pd.read_csv(data_path)
    df.columns = ["year", "prop_coverage"]
    df["prop_coverage"] = df["prop_coverage"] / 100
    df = df.sort_values("year", ascending=True)
    df.reset_index(inplace=True, drop=True)

    return df
