from pathlib import Path
import pandas as pd
from autumn.settings import INPUT_DATA_PATH

INPUT_DATA_PATH = Path(INPUT_DATA_PATH)

SERO_SURVEY_CSV = INPUT_DATA_PATH / "serotracker" / "serotracker_dataset.csv"
SERO_SURVEY_URL = (
    "https://raw.githubusercontent.com/serotracker/sars-cov-2-data/main/serotracker_dataset.csv"
)


def fetch_covid_serosurvey_data():
    pd.read_csv(SERO_SURVEY_URL).to_csv(SERO_SURVEY_CSV)
