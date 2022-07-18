import pandas as pd
from autumn.core.inputs.database import get_input_db
from autumn.settings.constants import COVID_BASE_DATETIME

TODAY = (pd.to_datetime("today").date() - COVID_BASE_DATETIME.date()).days


def get_percent_mc(iso3: str = None, col_name: str = "pct"):
    return get_survey_results(iso3, col_name, "survey_mask")


def get_percent_avoid_contact(iso3: str = None, col_name: str = "pct"):
    return get_survey_results(iso3, col_name, "survey_avoid_contact")


def get_survey_results(iso3, col_name, table_name):
    input_db = get_input_db()
    df = input_db.query(table_name, columns=["date_index", col_name], conditions={"iso_code": iso3})
    dates = df["date_index"].to_numpy()
    values = df[col_name].to_numpy()

    valid_dates = all(dates >= 1) and all(dates <= TODAY)
    valid_values = all(values >= 0) and all(values <= 1)

    if not valid_dates:
        raise AssertionError("Date index out of range")
    if not valid_values:
        raise AssertionError("Percentage values out of logical bounds")

    return dates, values
