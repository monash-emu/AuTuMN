from pathlib import Path

from autumn.settings import INPUT_DATA_PATH

INPUT_DATA_PATH = Path(INPUT_DATA_PATH)

SERO_SURVEY = INPUT_DATA_PATH / "sero-survey" / "sero-survey-prevelance.xlsx"


def fetch_covid_serosurvey_data():
    pass