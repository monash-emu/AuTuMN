import pandas as pd

from autumn.core.inputs.database import get_input_db


def get_mmr_testing_numbers():
    """
    Returns daily PCR test numbers for Myanmar
    """

    input_db = get_input_db()
    df = input_db.query(
        "covid_mmr",
        columns=["date_index", "tests"],
    )

    df.dropna(how="any", inplace=True)

    return pd.Series(df.tests.to_numpy(), index=df.date_index)


def base_mmr_adult_vacc_doses():
    # Slide 5 of Mya Yee Mon's PowerPoint sent on 12th November - applied to the 15+ population only

    """Will move this to inputs db"""
    times = [
        366,  # 1st Jan 2021
        393,  # 27th Jan
        499,  # 13th May
        522,  # 5th June
        599,  # 21st Aug
        606,  # 28th Aug
        613,  # 4th Sept
        620,  # 11th Sept
        627,  # 18th Sept
        634,  # 25th Sept
        641,  # 2nd Oct
        648,  # 9th Oct
        655,  # 16th Oct
        662,  # 23rd Oct
        665,  # 26th Oct
        670,  # 31st Oct
        678,  # 8th Nov
    ]

    values = [
        0,
        104865,
        1772177,
        1840758,
        4456857,
        4683410,
        4860264,
        4944654,
        5530365,
        7205913,
        8390746,
        9900823,
        11223285,
        12387573,
        12798322,
        13244996,
        13905795,
    ]

    return times, values
