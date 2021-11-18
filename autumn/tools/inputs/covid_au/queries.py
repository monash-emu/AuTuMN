from datetime import date, datetime

import numpy as np
import pandas as pd

from autumn.tools.inputs.database import get_input_db
from autumn.tools.utils.utils import apply_moving_average, COVID_BASE_DATETIME

COVID_BASE_DATE = date(2019, 12, 31)
TINY_NUMBER = 1e-6
VACC_COVERAGE_START_AGES = (0, 5, 12, 16, 20, 30, 40, 50, 60, 70, 80, 85)
VACC_COVERAGE_END_AGES = (4, 11, 15, 19, 29, 39, 49, 59, 69, 79, 84, 89)


def get_vic_testing_numbers():
    """
    Returns 7-day moving average of number of tests administered in Victoria.
    """
    input_db = get_input_db()
    df = input_db.query("covid_au", columns=["date", "tests"], conditions={"state_abbrev": "VIC"})
    date_str_to_int = lambda s: (datetime.strptime(s, "%Y-%m-%d") - COVID_BASE_DATETIME).days
    test_dates = df.date.apply(date_str_to_int).to_numpy()
    test_values = df.tests.to_numpy()
    avg_vals = np.array(apply_moving_average(test_values, 7)) + TINY_NUMBER
    return test_dates, avg_vals


def get_dhhs_testing_numbers(cluster: str = None):
    """
    Returns 7-day moving average of number of tests administered in Victoria.
    """
    input_db = get_input_db()

    if cluster is None:
        df = input_db.query("covid_dhhs_test", columns=["date", "test"])
        df = df.groupby("date", as_index=False).sum()
    else:
        df = input_db.query(
            "covid_dhhs_test", columns=["date", "test"], conditions={"cluster_name": cluster}
        )

    test_dates = (pd.to_datetime(df.date) - pd.datetime(2019, 12, 31)).dt.days.to_numpy()
    test_values = df.test.to_numpy()
    avg_vals = np.array(apply_moving_average(test_values, 7)) + TINY_NUMBER
    return test_dates, avg_vals


def get_historical_vac_coverage(
    cluster: str = None, start_age: int = 0, end_age: int = 89, dose=1
):
    """
    Returns vaccination coverage in Victorian health cluster.
    """
    input_db = get_input_db()

    cond_map = {
        "dosenumber": dose,
        "start_age>": start_age,
        "end_age<": end_age,
    }

    # Conditional to manage state or cluster
    cond_map = update_cond_map(cluster, cond_map)

    pop = get_pop(cluster, start_age, end_age, input_db)
    df = get_historical_vac_num(input_db, cond_map)

    # Total number of vaccinations per day
    df = df[["date_index", "n"]].groupby(["date_index"], as_index=False).sum()

    # Cumulative vaccination and coverage
    df["cml_n"] = df.n.cumsum()
    df["cml_coverage"] = df.cml_n / pop

    vac_dates = df.date_index.to_numpy()
    coverage_values = df.cml_coverage.to_numpy()
    avg_vals = np.array(apply_moving_average(coverage_values, 7)) + TINY_NUMBER
    return vac_dates, avg_vals

def update_cond_map(cluster, cond_map):
    if cluster is None:
        cluster = "Victoria"


def get_historical_vac_num(input_db, cond_map):
    df = input_db.query(
        "vic_2021", columns=["date_index", "n", "start_age", "end_age"], conditions=cond_map
    )

    return df


def get_modelled_vac_num(input_db, cond_map, dose):
    df = input_db.query(
        "vida_vac_model", columns=["date_index", dose, "start_age", "end_age"], conditions=cond_map
    )

    return df


def get_standard_vacc_coverage(iso3, age_group, age_pops):
    """
    Dummy function with arbitrary numbers.
    Would need to be populated with actual values for each age group.
    Note that with the current structure, this function must return something for each age group.
    """

    if iso3 == "LKA":
        times = [365, 395, 425, 455, 485, 515, 545, 575, 605, 635, 665, 695, 725]
        if int(age_group) < 15:
            coverage_values = [0.] * 13
        else:
            coverage_values = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.9, 0.9, 0.9]
    elif iso3 == "MMR":
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
            731,  # 31st December 2021
            1096,  # 31st December 2022
        ]

        # For the adult population
        if int(age_group) >= 15:
            adult_denominator = sum(age_pops[3:])

            # Slide 5 of Mya Yee Mon's PowerPoint sent on 12th November - applied to the 15+ population only
            at_least_one_dose = [
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

            # Convert doses to coverage
            coverage_values = [i_doses / adult_denominator for i_doses in at_least_one_dose]

            # Add future targets
            target_inflation_for_age = sum(age_pops) / adult_denominator
            target_all_age_coverage = [0.4, 0.7]
            target_adult_coverage = [target_inflation_for_age * i_cov for i_cov in target_all_age_coverage]
            assert all([0. <= i_cov <= 1. for i_cov in target_adult_coverage])
            coverage_values += target_adult_coverage

        # For the children, no vaccination
        else:
            coverage_values = [0.] * len(times)

    else:
        raise ValueError(f"No standard vaccination coverage information available for this country: {iso3}")

    assert len(times) == len(coverage_values)
    assert all((0. <= i_coverage <= 1. for i_coverage in coverage_values))
    return times, coverage_values


def get_both_vacc_coverage(cluster: str=None, start_age: int=0, end_age: int=89, dose="dose_1"):
    """
    Use the following function (get_modelled_vac_coverage) to get the same data out for both vaccines from data provided
    by Vida at the Department.
    Returns data in the same format as for the individual vaccines.
    """

    # Extract the data
    az_times, az_values = get_modelled_vac_coverage(cluster, start_age, end_age, vaccine="astra_zeneca", dose=dose)
    pfizer_times, pfizer_values = get_modelled_vac_coverage(cluster, start_age, end_age, vaccine="pfizer", dose=dose)

    assert all(az_times == pfizer_times), "Modelled coverage times are different for Pfizer and Astra-Zeneca"
    times = az_times
    both_values = az_values + pfizer_values
    return times, both_values


def get_modelled_vac_coverage(
    cluster: str = None, start_age: int = 0, end_age: int = 89, vaccine="pfizer", dose='dose_1'
):
    """Returns the vaccination coverage per Vida's DHHS model

    Args:
        cluster (str, optional): The DHHS clusters as all caps with underscores. Defaults to None.
        start_age (int, optional): Vida's start age brackets {0,5,12,16,20,30,40,50,60,70,80,85}. Defaults to 0.
        end_age (int, optional): Vida's end age brackets {4,11,15,19,29,39,49,59,69,79,84,89}. Defaults to 89.
        vaccine (str, optional): {pfizer, astra_zeneca}. Defaults to "pfizer".
        dose (str, optional): {dose_1, dose-2}. Defaults to 'dose_1'.

    Returns:
        tuple: two np.arrays of weekly dates and coverage
    """

    msg = f"Starting age not one available from modelled vaccination coverage database: {start_age}"
    assert start_age in VACC_COVERAGE_START_AGES, msg
    msg = f"Finishing age not one available from modelled vaccination coverage database: {end_age}"
    assert end_age in VACC_COVERAGE_END_AGES, msg
    msg = f"Starting age ({start_age}) vaccination coverage equal to or greater than finishing age ({end_age})"
    assert start_age < end_age, msg
    msg = f"Requested vaccine not available: {vaccine}"
    assert vaccine in ("pfizer", "astra_zeneca"), msg

    input_db = get_input_db()

    cond_map = {
        "vaccine_brand_name": vaccine,
        "start_age>": start_age, 
        "end_age<": end_age,
    }

    # Conditional to manage state or cluster
    cond_map = update_cond_map(cluster, cond_map)

    pop = vida_pop(cluster, start_age, end_age, input_db)
    df = get_modelled_vac_num(input_db, cond_map, dose)

    # Total number of vaccinations per day
    df = df[["date_index", dose]].groupby(["date_index"], as_index=False).sum()

    # Cumulative vaccination and coverage
    df[f"cml_{dose})"] = df[dose].cumsum()
    df["cml_coverage"] = df[f"cml_{dose})"] / pop

    vac_dates = df.date_index.to_numpy()
    coverage_values = df.cml_coverage.to_numpy()
    avg_vals = np.array(apply_moving_average(coverage_values, 7)) + TINY_NUMBER
    return vac_dates, avg_vals


def vida_pop(cluster, start_age, end_age, input_db):
    """Returns the denominator as per Vida's population numbers
    for a given health cluster and age band"""
    pop = input_db.query(
        "vida_pop",
        columns=["popn"],
        conditions={            
            "cluster_id": cluster,
            "start_age>": start_age,
            "end_age<": end_age,
        },
    )
    pop = pop.popn.sum()
    return pop


def get_pop(cluster, start_age, end_age, input_db):
    pop = input_db.query(
        "population",
        columns=["population"],
        conditions={
            "year": 2020,
            "region": cluster,
            "start_age>": start_age,
            "end_age<": end_age,
        },
    )
    pop = pop.population.sum()
    return pop


def update_cond_map(cluster, cond_map):
    if cluster is None:
        cluster = "Victoria"

    elif cluster is not None:
        cluster = cluster.upper()
        cond_map["cluster_id"] = cluster
    return cond_map


def get_yougov_date():
    """ Return the entire YouGov table for Victoria"""
    input_db = get_input_db()
    df = input_db.query("yougov_vic")
    return df
