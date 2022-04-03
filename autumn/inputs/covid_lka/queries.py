import numpy as np  

from autumn.inputs.database import get_input_db
from autumn.inputs.demography.queries import get_population_by_agegroup
from autumn.models.covid_19.parameters import TimeSeries


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
    test_dates = df.date_index.to_numpy()
    test_values = df.Sri_Lanka_PCR_tests_done.to_numpy()

    return test_dates, test_values

def get_lka_vac_coverage(age_group, age_pops=None, params=None):
    """ Provides vaccination coverage for a given age.
    It is assumed all ages above 14 have uniform coverage"""

    vaccinated_population = get_population_by_agegroup([0,15], "LKA")[1] # 15+ pop
    input_db = get_input_db()

    df = input_db.query("covid_lka",
    columns = ["date_index", "Sri_Lanka_COVID19_Vaccination_1st_Doses_Total"]
    )
    df.rename(columns={"Sri_Lanka_COVID19_Vaccination_1st_Doses_Total": "cml_vac_dose_1"}, inplace =True)
    df.dropna(how="any", inplace=True)

    df["cml_coverage"] = df.cml_vac_dose_1 / vaccinated_population
    
    times = df.date_index.to_numpy()

    if int(age_group) < 15:
        coverage_values = (df.cml_coverage * 0).to_numpy()
    else:
        coverage_values = df.cml_coverage.to_numpy()

    coverage_too_large = any(coverage_values >= .99)
    unequal_len = len(times) != len(coverage_values)
    if any([coverage_too_large,unequal_len]):
        AssertionError("Unrealistic coverage")
    
    return TimeSeries(times=times, values=coverage_values)




