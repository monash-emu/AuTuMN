from pathlib import Path

from autumn.settings import INPUT_DATA_PATH

INPUT_DATA_PATH = Path(INPUT_DATA_PATH)

GISAID_VOC = INPUT_DATA_PATH / "gisaid-voc" / "gisaid_voc_data.xlsx"


def fetch_covid_gisaid_data():
    pass
