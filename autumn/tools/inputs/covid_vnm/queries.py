from autumn.tools.inputs.database import get_input_db


def get_vnm_testing_numbers(subregion):
    """
    Returns daily PCR test numbers for Vietnam
    """

    input_db = get_input_db()
    df = input_db.query(
        "covid_vnm",
        columns=["date_index", "daily_test", "region"],
    )
    df.dropna(how="any", inplace=True)
    df = df[df.region.str.lower() == subregion.lower()]
    test_dates = df.date_index.to_numpy()
    test_values = df.daily_test.to_numpy()

    return test_dates, test_values
