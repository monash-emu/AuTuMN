import os
from pathlib import Path

from autumn.settings import INPUT_DATA_PATH

INPUT_DATA_PATH = Path(INPUT_DATA_PATH)

TB_CAMAU_DIRPATH = INPUT_DATA_PATH / "tb_camau"
TB_CAMAU_CSV_PATH = TB_CAMAU_DIRPATH / "cbr-cdr-camau.csv"


def fetch_tb_camau_data():
    pass
