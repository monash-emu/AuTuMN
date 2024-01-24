"""
Read demography data from input database.
"""
from typing import List

import numpy as np
import pandas as pd

INF = float("inf")

from autumn.settings import INPUT_DATA_PATH
from pathlib import Path
INPUT_DATA_PATH = Path(INPUT_DATA_PATH)

def get_input_df(file, columns=None):
    return pd.read_parquet(INPUT_DATA_PATH / file, columns=columns)

def _get_death_rates(country_iso_code: str):
    if country_iso_code in MAPPING_ISO_CODE:
        country_iso_code = MAPPING_ISO_CODE[country_iso_code]
    input_db = get_input_db()
    death_df = input_db.query(
        "deaths",
        conditions={"iso3": country_iso_code},
    )
    pop_df = input_db.query(
        "population",
        conditions={
            "iso3": country_iso_code,
            "region": None,
        },
    )
    # Calculate mean year and time period
    death_df["mean_year"] = (death_df["start_year"] + death_df["end_year"]) / 2
    death_df["period"] = death_df["end_year"] - death_df["start_year"]

    # Combine population and total death data so we can calulate death rate.
    # Throws away data for population over 100 y.o.
    rate_df = pd.merge(
        death_df, pop_df, left_on=["start_year", "start_age"], right_on=["year", "start_age"]
    )

    rate_df["population"] = rate_df["population"].where(rate_df["population"] > 0.0, 1.0)

    # Calculate death rate.
    rate_df["death_rate"] = rate_df["death_count"] / (rate_df["population"] * rate_df["period"])

    cols = ["mean_year", "start_age", "death_rate"]
    rate_df = rate_df.drop(columns=[c for c in rate_df.columns if c not in cols])
    rate_df = rate_df.sort_values(["mean_year", "start_age"])
    return rate_df


def _get_life_expectancy(country_iso_code: str):
    input_db = get_input_db()
    expectancy_df = input_db.query("life_expectancy", conditions={"iso3": country_iso_code})

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
    age_breakpoints = _check_age_breakpoints(age_breakpoints)
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
    age_breakpoints = _check_age_breakpoints(age_breakpoints)
    life_expectancy_df = _get_life_expectancy(country_iso_code)
    years = life_expectancy_df["mean_year"].unique().tolist()
    orig_ages = life_expectancy_df["start_age"].unique().tolist()
    year_step = 5
    year_expectancy = {}
    for year in years:
        orig_expectancy = life_expectancy_df[life_expectancy_df["mean_year"] == year][
            "life_expectancy"
        ].tolist()
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
    if country_name.lower() in MAPPING_COUNTRY_ISO:
        return MAPPING_COUNTRY_ISO[country_name.lower()]
    elif country_name in MAPPING_COUNTRY_ISO:
        return MAPPING_COUNTRY_ISO[country_name]
    else:
        input_db = get_input_db()
        country_df = input_db.query("countries", conditions={"country": country_name})
        results = country_df["iso3"].tolist()
        if results:
            return results[0]
        else:
            raise ValueError(f"Country name {country_name} not found")

def get_gisaid_country_name_from_iso3(iso3):
    input_db = get_input_db()
    gisaid_countries = input_db.query(table_name='gisaid', conditions= {}).Country.unique()
    
    keep_going = True
    i = -1
    while keep_going:
        i += 1
        c = gisaid_countries[i]
        try:
            if get_iso3_from_country_name(c) == iso3:
                return c
        except:
            keep_going = True

def get_crude_birth_rate(country_iso_code: str):
    """
    Gets crude birth rate over time for a given country.
    Returns a list of birth rates and a list of years.
    """
    if country_iso_code in MAPPING_ISO_CODE:
        country_iso_code = MAPPING_ISO_CODE[country_iso_code]
    input_db = get_input_db()
    birth_df = input_db.query("birth_rates", conditions={"iso3": country_iso_code})
    birth_df = birth_df.sort_values(["mean_year"])
    return birth_df["birth_rate"], birth_df["mean_year"]


def get_population_by_agegroup(
    age_breakpoints: List[int], country_iso_code: str, year: int = 2020
):
    """
    Find population for age bins.
    Returns a list of ints, each item being the population for that age bracket.
    """
    age_breakpoints = _check_age_breakpoints(age_breakpoints)



    # Work out the year that is nearest to the requested year, among available years.
    
    pop_df = get_input_df("population.parquet")
    pop_df = pop_df[pop_df["iso3"] == country_iso_code]
    
    available_years = pop_df["year"].unique()
    nearest_year = min(available_years, key=lambda x: abs(x-year))

    pop_df = pop_df[pop_df["year"] == nearest_year]
    pop_df = pop_df.sort_values(["start_age"])

    pop_df_with_data = pop_df.dropna(subset=["population"])
    orig_ages = pop_df_with_data["start_age"].tolist()
    orig_pop = pop_df_with_data["population"].tolist()
    assert len(orig_ages) == len(orig_pop)
    population = downsample_quantity(orig_pop, orig_ages, age_breakpoints)

    return [int(p) for p in population]


def convert_ifr_agegroups(raw_ifr_props: list, iso3: str, pop_region: str, pop_year: int) -> list:
    """
    Converts the IFRs from the age groups they were provided in to the ones needed for the model.
    """

    # Work out the proportion of 80+ years old among the 75+ population
    elderly_populations = get_population_by_agegroup([0, 75, 80], iso3, year=pop_year)
    prop_over_80 = elderly_populations[2] / sum(elderly_populations[1:])

    # Calculate 75+ age bracket as weighted average between 75-79 and 80+
    ifr_over75 = raw_ifr_props[-1] * prop_over_80 + raw_ifr_props[-2] * (1. - prop_over_80)
    return [*raw_ifr_props[:-2], ifr_over75]


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
                weights[i_o, i_n] = (min(new_end, orig_end) - new_start) / (orig_end - orig_start)
            elif new_start < orig_start and new_end >= orig_end:
                # New bin encompasses old bin, add the whole thing
                weights[i_o, i_n] = 1
            elif orig_start < new_end < orig_end:
                # New bin ends inside an old bin, take a fraction of the start.
                weights[i_o, i_n] = (new_end - orig_start) / (orig_end - orig_start)

    return weights


def _check_age_breakpoints(age_breakpoints: List[str]) -> List[float]:
    age_breakpoints = [int(s) for s in age_breakpoints]
    assert age_breakpoints == sorted(age_breakpoints)
    assert age_breakpoints[0] == 0
    return age_breakpoints


"""
Movement and mobility
"""

from datetime import datetime
import pandas as pd

def get_mobility_data(country_iso_code: str, base_date: datetime):
    """
    Get daily Google mobility data for locations, for a given country.
    Times are in days since a given base date.
    The location map parameter transforms Google Mobility locations into Autumn-friendly locations.
    """
    mob_df = get_google_mobility(country_iso_code, base_date)
    mov_df = get_movement_data(country_iso_code)

    merge_how = "right" if len(mob_df) > 0 else "left"
    mob_df = pd.merge(mov_df, mob_df, on="date", how=merge_how)

    mob_df["iso3"] = country_iso_code
    mob_df = mob_df.sort_values(["date"])
    mob_df = mob_df[mob_df["date"] >= base_date]

    days = mob_df["date"].apply(lambda d: (d - base_date).days).tolist()
    return mob_df, days


def get_google_mobility(country_iso_code, base_date):
    #conditions = {"iso3": country_iso_code, "region": region or None}
    #mob_df = input_db.query("mobility", conditions=conditions)
    mob_df = get_input_df("mobility/mobility.parquet")
    mob_df = mob_df[mob_df["iso3"]==country_iso_code]
    mob_df["date"] = pd.to_datetime(mob_df["date"], format="%Y-%m-%d")
    mob_df = mob_df.sort_values(["date"])
    mob_df = mob_df[mob_df["date"] >= base_date]
    return mob_df

def get_movement_data(country_iso_code: str):
    """
    Get daily Facebook mobility data for locations, for a given country.
    Times are in days since a given base date.

    """
    mov_df = get_input_df("mobility/movement.parquet")
    mov_df = mov_df[mov_df["country"] == country_iso_code]

    mov_df["date"] = pd.to_datetime(mov_df["date"], format="%Y-%m-%d")
    mov_df = mov_df.sort_values(["date"])
    mov_df = mov_df.rename(
        columns={
            "all_day_bing_tiles_visited_relative_change": "tiles_visited",
            "all_day_ratio_single_tile_users": "single_tile",
        }
    )
    # mov_df = mov_df[mov_df["date"] >= base_date]
    mov_df["tiles_visited"] += 1
    mov_df["single_tile"] += 1
    mov_df = mov_df.groupby(["date", "date_index"], as_index=False).mean()

    return mov_df
