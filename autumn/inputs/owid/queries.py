from datetime import date, datetime

from autumn.inputs.database import get_input_db
from autumn.tool_kit.utils import apply_moving_average
from autumn.inputs.covid_au.queries import COVID_BASE_DATETIME


def get_international_testing_numbers(iso3):
    """
    Returns 7-day moving average of number of tests administered in Victoria.
    """

    input_db = get_input_db()
    df = input_db.query("owid", columns=["date", "new_tests"], conditions={"iso_code": iso3})
    df_with_data = df.dropna()  # dropna default behaviour is to drop entire row if any nas
    date_str_to_int = lambda s: (datetime.strptime(s, "%d/%m/%Y") - COVID_BASE_DATETIME).days
    test_dates = list(df_with_data.date.apply(date_str_to_int).to_numpy())
    test_numbers = list(df_with_data.loc[:, "new_tests"])
    return test_dates, test_numbers
