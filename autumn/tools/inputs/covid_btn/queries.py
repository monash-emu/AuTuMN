from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd

from autumn.tools.inputs.database import get_input_db
from autumn.tools.inputs.demography.queries import get_population_by_agegroup
from autumn.tools.utils.utils import check_list_increasing

TINY_NUMBER = 1e-6


def get_btn_testing_numbers(subregion: Optional[str]):
    """
    Returns number of tests administered in Bhutan or Thimphu.
    """

    subregion = "Bhutan" if subregion is False else subregion

    cond_map = {
        "region": subregion,
    }

    input_db = get_input_db()
    df = input_db.query(
        "covid_btn_test", columns=["date_index", "total_tests"], conditions=cond_map
    )
    df.dropna(inplace=True)
    test_dates = df.date_index.to_numpy()
    values = df["total_tests"].to_numpy() + TINY_NUMBER

    return test_dates, values
