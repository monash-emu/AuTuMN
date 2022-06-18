from datetime import datetime
import pandas as pd

from autumn.core.inputs.database import get_input_db
from autumn.settings.constants import COVID_BASE_DATETIME


def get_international_owid_numbers(iso3, variable="new_tests"):
    input_db = get_input_db()
    df = input_db.query("owid", columns=["date", variable], conditions={"iso_code": iso3})
    df_with_data = df.dropna()  # dropna default behaviour is to drop entire row if any nas
    date_str_to_int = lambda s: (datetime.strptime(s, "%Y-%m-%d") - COVID_BASE_DATETIME).days
    dates = df_with_data.date.apply(date_str_to_int)
    numbers = list(df_with_data.loc[:, variable])
    return pd.Series(numbers, index=dates)

