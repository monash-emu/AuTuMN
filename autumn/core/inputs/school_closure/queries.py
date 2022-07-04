from datetime import date, datetime

import pandas as pd
from autumn.core.inputs.database import get_input_db


def get_school_closure(col_list: list) -> pd.DataFrame:
    """
    Returns the columns of interest of the UNESCO school closure table
    Args:
        col_list (list): List of columns of interest per school_closure.csv file.
    Returns:
        pd.Dataframe: A Pandas data frame of columns of interest
    """
    input_db = get_input_db()
    col_list= [each.lower().replace(" ","_") for each in col_list]
    df = input_db.query("school_closure", columns=["date_index", *col_list])
    df.set_index('date_index', inplace=True)

    return df
