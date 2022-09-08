import pandas as pd

from autumn.core.inputs.database import get_input_db
from autumn.core.inputs.demography.queries import get_population_by_agegroup


def get_lka_testing_numbers():
    """
    Returns daily PCR test numbers for Sri lanka
    """

    input_db = get_input_db()
    df = input_db.query(
        "covid_lka",
        columns=["date_index", "Sri_Lanka_PCR_tests_done"],
    )
    df.dropna(how="any", inplace=True)

    return pd.Series(df.Sri_Lanka_PCR_tests_done.to_numpy(), index=df.date_index)


def get_lka_vac_coverage(age_group, age_pops=None, params=None):
    """Provides vaccination coverage for a given age.
    It is assumed all ages above 14 have uniform coverage"""

    vaccinated_population = get_population_by_agegroup([0,11], "LKA")[1] # 10+ pop

    input_db = get_input_db()

    df = input_db.query(
        "covid_lka", columns=["date_index", "Sri_Lanka_COVID19_Vaccination_1st_Doses_Total"]
    )
    df.rename(
        columns={"Sri_Lanka_COVID19_Vaccination_1st_Doses_Total": "cml_vac_dose_1"}, inplace=True
    )
    df.dropna(how="any", inplace=True)

    df.loc[df["date_index"] == 452, "cml_vac_dose_1"] = 0  # Have to make the first date zero!
    df.sort_values("date_index", inplace=True)
    df.reset_index(drop=True, inplace=True)

    df["cml_coverage"] = df.cml_vac_dose_1 / vaccinated_population

    times = df.date_index.to_numpy()

    if int(age_group) < 0:
        coverage_values = (df.cml_coverage * 0).tolist()
    else:
        coverage_values = df.cml_coverage.tolist()


    coverage_too_large = any([each >= 0.99 for each in coverage_values])

    unequal_len = len(times) != len(coverage_values)
    if any([coverage_too_large, unequal_len]):
        AssertionError("Unrealistic coverage")

    return TimeSeries(times=times, values=coverage_values)
