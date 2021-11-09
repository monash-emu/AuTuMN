from autumn.tools.inputs.database import get_input_db
from autumn.tools.utils.utils import apply_moving_average, COVID_BASE_DATETIME

import numpy as np
import pandas as pd

TODAY = (pd.to_datetime("today").date() - COVID_BASE_DATETIME.date()).days


def get_percent_mc(iso3: str = None, col_name: str = None):
    input_db = get_input_db()
    df = input_db.query(
        "survey_mask", columns=["date_index", col_name], conditions={"iso_code": iso3}
    )
    mask_dates = df["date_index"].to_numpy()
    mask_values = df["percent_mc"].to_numpy()

    if not all(1 <= mask_dates) and all(mask_dates <= TODAY):
        raise AssertionError("Date index out of range")
    if not all(0 <= mask_values) and all(mask_values <= 1):
        raise AssertionError("Percentage values out of logical bounds")

    return mask_dates, mask_values
