"""
Read demography data from input database.
"""
from typing import List
from functools import lru_cache

import numpy as np
import pandas as pd

from autumn.inputs.database import get_input_db

INF = float("inf")


def _get_death_rates(country_iso_code: str):
    input_db = get_input_db()
    death_df = input_db.query("deaths", conditions=[f"iso3='{country_iso_code}'"],)
    pop_df = input_db.query(
        "population", conditions=[f"iso3='{country_iso_code}'", "region IS NULL",],
    )

    # Calculate mean year and time period
    death_df["mean_year"] = (death_df["start_year"] + death_df["end_year"]) / 2
    death_df["period"] = death_df["end_year"] - death_df["start_year"]

    # Combine population and total death data so we can calulate death rate.
    # Throws away data for population over 100 y.o.
    rate_df = pd.merge(
        death_df, pop_df, left_on=["start_year", "start_age"], right_on=["year", "start_age"]
    )

    # Calculate death rate.
    rate_df["death_rate"] = rate_df["death_count"] / (rate_df["population"] * rate_df["period"])

    cols = ["mean_year", "start_age", "death_rate"]
    rate_df = rate_df.drop(columns=[c for c in rate_df.columns if c not in cols])
    rate_df = rate_df.sort_values(["mean_year", "start_age"])
    return rate_df


def _get_life_expectancy(country_iso_code: str):
    input_db = get_input_db()
    expectancy_df = input_db.query("life_expectancy", conditions=[f"iso3='{country_iso_code}'"],)

    # Calculate mean year
    expectancy_df["mean_year"] = (expectancy_df["start_year"] + expectancy_df["end_year"]) / 2

    cols = ["mean_year", "start_age", "life_expectancy"]
    expectancy_df = expectancy_df.drop(columns=[c for c in expectancy_df.columns if c not in cols])
    expectancy_df = expectancy_df.sort_values(["mean_year", "start_age"])
    return expectancy_df


def get_death_rates_by_agegroup(age_breakpoints: List[float], country_iso_code: str):
    """
    Find death rates from UN data that are specific to the age groups provided.
    Returns a list of death rates and a list of years.
    """
    assert age_breakpoints == sorted(age_breakpoints)
    assert age_breakpoints[0] == 0
    input_db = get_input_db()
    rate_df = _get_death_rates(country_iso_code)
    years = rate_df["mean_year"].unique().tolist()
    orig_ages = rate_df["start_age"].unique().tolist()
    year_step = 5
    year_rates = {}
    for year in years:
        orig_rates = rate_df[rate_df["mean_year"] == year]["death_rate"].tolist()
        new_rates = downsample_rate(orig_rates, orig_ages, year_step, age_breakpoints)
        year_rates[year] = new_rates

    death_rates_by_agegroup = {}
    for i, age in enumerate(age_breakpoints):
        death_rates_by_agegroup[age] = [year_rates[y][i] for y in years]

    return death_rates_by_agegroup, years


def get_life_expectancy_by_agegroup(age_breakpoints: List[float], country_iso_code: str):
    """
    Find life expectancy from UN data that are specific to the age groups provided.
    Returns a list of life expectancy and a list of years.
    """
    assert age_breakpoints == sorted(age_breakpoints)
    assert age_breakpoints[0] == 0
    life_expectancy_df = _get_life_expectancy(country_iso_code)
    years = life_expectancy_df["mean_year"].unique().tolist()
    orig_ages = life_expectancy_df["start_age"].unique().tolist()
    year_step = 5
    year_expectancy = {}
    for year in years:
        orig_expectancy = life_expectancy_df[life_expectancy_df["mean_year"] == year]["life_expectancy"].tolist()
        new_expectancy = downsample_rate(orig_expectancy, orig_ages, year_step, age_breakpoints)
        year_expectancy[year] = new_expectancy

    life_expectancy_by_agegroup = {}
    for i, age in enumerate(age_breakpoints):
        life_expectancy_by_agegroup[age] = [year_expectancy[y][i] for y in years]

    return life_expectancy_by_agegroup, years


def get_iso3_from_country_name(country_name: str):
    """
    Return the iso3 code matching with a given country name.
    """
    input_db = get_input_db()
    country_df = input_db.query("countries", conditions=[f"country='{country_name}'"])
    results = country_df["iso3"].tolist()
    if results:
        return results[0]
    else:
        raise ValueError(f"Country name {country_name} not found")


def get_crude_birth_rate(country_iso_code: str):
    """
    Gets crude birth rate over time for a given country.
    Returns a list of birth rates and a list of years.
    """
    input_db = get_input_db()
    birth_df = input_db.query("birth_rates", conditions=[f"iso3='{country_iso_code}'"])
    birth_df = birth_df.sort_values(["mean_year"])
    return birth_df["birth_rate"].tolist(), birth_df["mean_year"].tolist()


def get_population_by_agegroup(
    age_breakpoints: List[float], country_iso_code: str, region: str = None, year: int = 2020
):
    """
    Find population for age bins.
    Returns a list of ints, each item being the population for that age bracket.
    """
    assert age_breakpoints == sorted(age_breakpoints)
    assert age_breakpoints[0] == 0
    input_db = get_input_db()
    pop_df = input_db.query(
        "population",
        conditions=[
            f"iso3='{country_iso_code}'",
            f"year={year}",
            f"region='{region}'" if region else "region IS NULL",
        ],
    )
    pop_df = pop_df.sort_values(["start_age"])
    orig_ages = pop_df["start_age"].tolist()
    orig_pop = pop_df["population"].tolist()
    assert len(orig_ages) == len(orig_pop)
    population = downsample_quantity(orig_pop, orig_ages, age_breakpoints)
    return [int(p) for p in population]


def downsample_rate(
    orig_rates: List[float], orig_bins: List[float], orig_step: float, new_bins: List[float]
):
    """
    Downsample original rates from their current bins to new bins
    Assume new bins are smaller than, or equal to, the original bins.
    Requires that original values are equispaced by `orig_step` amount.
    """
    num_orig_bins = len(orig_bins)
    num_new_bins = len(new_bins)
    weights = get_bin_weights(orig_bins, new_bins)
    new_rates = [0 for _ in range(num_new_bins)]

    orig_rates = np.array(orig_rates)
    for i_n in range(num_new_bins):
        time_chunks = np.zeros(num_orig_bins)
        for i_o in range(num_orig_bins):
            time_chunks[i_o] = weights[i_o, i_n] * orig_step

        new_rates[i_n] = (orig_rates * time_chunks).sum() / time_chunks.sum()

    return new_rates


def downsample_quantity(orig_vals: List[float], orig_bins: List[float], new_bins: List[float]):
    """
    Downsample original values from their current bins to new bins
    Assume new bins are smaller than, or equal to, the original bins
    """
    num_orig_bins = len(orig_bins)
    num_new_bins = len(new_bins)
    weights = get_bin_weights(orig_bins, new_bins)
    new_vals = [0 for _ in range(num_new_bins)]
    for i_n in range(num_new_bins):
        for i_o in range(num_orig_bins):
            new_vals[i_n] += weights[i_o, i_n] * orig_vals[i_o]

    assert sum(orig_vals) - sum(new_vals) < 1e-3
    return new_vals


def get_bin_weights(orig_bins: List[float], new_bins: List[float]):
    """
    Gets 2D weight matrix for moving from orig bins to new bins. 
    """
    num_orig_bins = len(orig_bins)
    num_new_bins = len(new_bins)
    weights = np.zeros([num_orig_bins, num_new_bins])
    for i_n, new_start in enumerate(new_bins):
        # Find the new bin end
        if i_n == num_new_bins - 1:
            new_end = INF
        else:
            new_end = new_bins[i_n + 1]

        # Loop through all old bins, take matching proportion
        for i_o, orig_start in enumerate(orig_bins):
            # Find the orig bin end
            if i_o == len(orig_bins) - 1:
                orig_end = INF
            else:
                orig_end = orig_bins[i_o + 1]

            is_new_bin_inside_old_one = new_start > orig_start and new_end < orig_end
            assert not is_new_bin_inside_old_one, "New bin inside old bin"

            if orig_end == INF and new_end == INF:
                # Final bins, add everything
                assert new_start <= orig_start, "Cannot slice up infinity"
                weights[i_o, i_n] = 1
            elif orig_start <= new_start < orig_end:
                # New bin starts at start, or half way through an old bin
                # We get a fraction of the end of the bin
                weights[i_o, i_n] = (orig_end - new_start) / (orig_end - orig_start)
            elif new_start < orig_start and new_end >= orig_end:
                # New bin encompasses old bin, add the whole thing
                weights[i_o, i_n] = 1
            elif orig_start < new_end < orig_end:
                # New bin ends inside an old bin, take a fraction of the start.
                weights[i_o, i_n] = (new_end - orig_start) / (orig_end - orig_start)

    return weights
