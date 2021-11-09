from autumn.tools.inputs.database import get_input_db
from autumn.tools.utils.utils import apply_moving_average, COVID_BASE_DATETIME

import numpy as np
import pandas as pd

TODAY = (pd.to_datetime("today").date() - COVID_BASE_DATETIME.date()).days


def get_percent_mc(iso3: str = None, col_name: str = "percent_mc"):
    input_db = get_input_db()
    df = input_db.query(
        "survey_mask", columns=["date_index", col_name], conditions={"iso_code": iso3}
    )
    mask_dates = df["date_index"].to_numpy()
    mask_values = df[col_name].to_numpy()

    valid_dates = all(1 <= mask_dates) and all(mask_dates <= TODAY)
    valid_values = all(0 <= mask_values) and all(mask_values <= 1)

    if not valid_dates:
        raise AssertionError("Date index out of range")
    if not valid_values:
        raise AssertionError("Percentage values out of logical bounds")

    return mask_dates, mask_values
