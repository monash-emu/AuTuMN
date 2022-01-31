"""
Preprocess demography data so it is included in the inputs database
"""
import os

import pandas as pd
from sqlalchemy import column

from autumn.tools.db import Database
from autumn.settings import INPUT_DATA_PATH, region

POP_DIRPATH = os.path.join(INPUT_DATA_PATH, "world-population")
BGD_POP = os.path.join(INPUT_DATA_PATH, "covid_bgd", "BD_PopulationProjection_2021_BBS.xlsx")
ROHINGYA_POP = os.path.join(
    INPUT_DATA_PATH, "covid_bgd", "UNHCR Population Factsheet  Block Level Data.xlsx"
)


def preprocess_demography(input_db: Database):
    loc_df = read_location_df()
    pop_df = read_population_df(loc_df)
    birth_df = read_crude_birth_df(loc_df)
    death_df = read_death_df(loc_df)
    expect_df = read_life_expectancy_df(loc_df)
    input_db.dump_df("countries", loc_df)
    input_db.dump_df("population", pop_df)
    input_db.dump_df("birth_rates", birth_df)
    input_db.dump_df("deaths", death_df)
    input_db.dump_df("life_expectancy", expect_df)
    return loc_df


def read_life_expectancy_df(loc_df: pd.DataFrame):
    """
    Read in life expectancy by age.
    """
    expect_path = os.path.join(
        POP_DIRPATH, "WPP2019_MORT_F16_1_LIFE_EXPECTANCY_BY_AGE_BOTH_SEXES.xlsx"
    )
    expect_df = pd.read_excel(
        pd.ExcelFile(expect_path),
        header=16,
        index_col=0,
        sheet_name="ESTIMATES",
    )

    # Rename columns to a simpler format
    expect_df.rename(columns={"Country code": "country_code"}, inplace=True)

    # Drop unwanted columns
    expect_cols = [str(5 * i) for i in range(20)] + ["100+"]
    cols = ["country_code", "Period", *expect_cols]
    expect_df = expect_df.drop(columns=[c for c in expect_df.columns if c not in cols])

    # Add in iso3 info from location dataframe, drop country code info
    expect_df = pd.merge(loc_df, expect_df, left_on="country_code", right_on="country_code")
    expect_df = expect_df.drop(columns=["country_code"])

    # Split period into start / end years
    expect_df["start_year"] = expect_df["Period"].apply(lambda s: int(s.split("-")[0]))
    expect_df["end_year"] = expect_df["Period"].apply(lambda s: int(s.split("-")[1]))
    expect_df = expect_df.drop(columns=["Period"])

    # Unpivot data so each age group gets its own row
    expect_df = expect_df.melt(
        id_vars=["country", "iso3", "start_year", "end_year"], value_vars=expect_cols
    )
    expect_df.rename(columns={"value": "life_expectancy"}, inplace=True)

    def label_ages(age_str):
        age = int(age_str.replace("+", ""))
        return [age, age + 4]

    ages_df = pd.DataFrame(
        [label_ages(age_str) for age_str in expect_df.variable],
        columns=("start_age", "end_age"),
    )
    expect_df = expect_df.join(ages_df)
    expect_df = expect_df.drop(columns="variable")

    # Ensure all numbers are actually numbers
    numeric_cols = ["start_year", "end_year", "start_age", "end_age", "life_expectancy"]
    expect_df[numeric_cols] = expect_df[numeric_cols].apply(pd.to_numeric)
    expect_df = expect_df.sort_values(["iso3", "start_year", "start_age"])
    return expect_df


def read_death_df(loc_df: pd.DataFrame):
    """
    Read in absolute number of deaths for a given time period,
    broken up by age bracket.
    """
    death_path = os.path.join(POP_DIRPATH, "WPP2019_MORT_F04_1_DEATHS_BY_AGE_BOTH_SEXES.xlsx")
    death_df = pd.read_excel(
        pd.ExcelFile(death_path),
        header=16,
        index_col=0,
        sheet_name="ESTIMATES",
    )

    # Rename columns to a simpler format
    death_df.rename(columns={"Country code": "country_code"}, inplace=True)

    # Drop unwanted columns
    deathrate_cols = [c for c in death_df.columns if "-" in c or c.endswith("+")]
    cols = ["country_code", "Period", *deathrate_cols]
    death_df = death_df.drop(columns=[c for c in death_df.columns if c not in cols])

    # Add in iso3 info from location dataframe, drop country code info
    death_df = pd.merge(loc_df, death_df, left_on="country_code", right_on="country_code")
    death_df = death_df.drop(columns=["country_code"])

    # Split period into start / end years
    death_df["start_year"] = death_df["Period"].apply(lambda s: int(s.split("-")[0]))
    death_df["end_year"] = death_df["Period"].apply(lambda s: int(s.split("-")[1]))
    death_df = death_df.drop(columns=["Period"])

    # Unpivot data so each age group gets its own row
    death_df = death_df.melt(
        id_vars=["country", "iso3", "start_year", "end_year"], value_vars=deathrate_cols
    )
    death_df.rename(columns={"value": "death_count"}, inplace=True)

    def label_ages(age_str):
        if age_str.endswith("+"):
            return (int(age_str.replace("+", "")), None)
        else:
            return [int(s) for s in age_str.split("-")]

    ages_df = pd.DataFrame(
        [label_ages(age_str) for age_str in death_df.variable],
        columns=("start_age", "end_age"),
    )
    death_df = death_df.join(ages_df)
    death_df = death_df.drop(columns="variable")

    # Ensure all numbers are actually numbers
    numeric_cols = ["start_year", "end_year", "start_age", "end_age", "death_count"]
    death_df[numeric_cols] = death_df[numeric_cols].apply(pd.to_numeric)
    death_df = death_df.sort_values(["iso3", "start_year", "start_age"])

    # Multiply population by 1000 to get actual numbers
    death_df["death_count"] = 1e3 * death_df["death_count"]

    return death_df


def read_crude_birth_df(loc_df: pd.DataFrame):
    """
    Read in births per 1000 people for a given time period,
    """
    birth_rate_path = os.path.join(POP_DIRPATH, "WPP2019_FERT_F03_CRUDE_BIRTH_RATE.xlsx")
    birth_df = pd.read_excel(
        pd.ExcelFile(birth_rate_path),
        header=16,
        index_col=0,
        sheet_name="ESTIMATES",
    )

    # Rename columns to a simpler format
    birth_df.rename(columns={"Country code": "country_code"}, inplace=True)

    # Drop unwanted columns
    birthrate_cols = [c for c in birth_df.columns if "-" in c]
    cols = ["country_code", *birthrate_cols]
    birth_df = birth_df.drop(columns=[c for c in birth_df.columns if c not in cols])

    # Add in iso3 info from location dataframe, drop country code info
    birth_df = pd.merge(loc_df, birth_df, left_on="country_code", right_on="country_code")
    birth_df = birth_df.drop(columns=["country_code"])

    # Unpivot data so each age group gets its own row
    birth_df = birth_df.melt(id_vars=["country", "iso3"], value_vars=birthrate_cols)
    birth_df.rename(columns={"value": "birth_rate"}, inplace=True)
    variables = ((int(p) for p in s.split("-")) for s in birth_df.variable)

    def label_times(start_time, end_time):
        return (
            start_time,
            end_time,
            (start_time + end_time) / 2,
        )

    times_df = pd.DataFrame(
        [label_times(*times) for times in variables],
        columns=("start_year", "end_year", "mean_year"),
    )
    birth_df = birth_df.join(times_df)
    birth_df = birth_df.drop(columns="variable")

    # Ensure all numbers are actually nummbers
    numeric_cols = ["birth_rate", "start_year", "end_year", "mean_year"]
    birth_df[numeric_cols] = birth_df[numeric_cols].apply(pd.to_numeric)

    birth_df = birth_df.sort_values(["iso3", "start_year"])
    return birth_df


def read_population_df(loc_df: pd.DataFrame):
    """
    Read UN country population estimates

    Read in absolute number of people for a given time period,
    broken up by age bracket.
    """
    region_pop_path = os.path.join(POP_DIRPATH, "subregions.csv")
    country_pop_path = os.path.join(
        POP_DIRPATH, "WPP2019_POP_F07_1_POPULATION_BY_AGE_BOTH_SEXES.xlsx"
    )
    region_df = pd.read_csv(region_pop_path)
    pop_df = pd.read_excel(
        pd.ExcelFile(country_pop_path),
        header=16,
        index_col=0,
        sheet_name="ESTIMATES",
    )

    # Rename columns to a simpler format
    rename_cols = {
        "Reference date (as of 1 July)": "year",
        "Country code": "country_code",
    }
    pop_df.rename(columns=rename_cols, inplace=True)

    # Drop unwanted columns
    agegroup_cols = [c for c in pop_df.columns if "-" in c or c.endswith("+")]
    cols = ["country_code", "year", *agegroup_cols]
    pop_df = pop_df.drop(columns=[c for c in pop_df.columns if c not in cols])

    # Add in iso3 info from location dataframe, drop country code info
    pop_df = pd.merge(loc_df, pop_df, left_on="country_code", right_on="country_code")
    pop_df = pop_df.drop(columns=["country_code"])

    # Add empty region column so we can add cities, states etc
    pop_df.insert(2, "region", [None for _ in range(len(pop_df))])
    assert (pop_df.columns == region_df.columns).all(), "Column mismatch"

    # Add in region data
    pop_df = pd.concat([region_df, pop_df], ignore_index=True)

    # Unpivot data so each age group gets its own row
    pop_df = pop_df.melt(id_vars=["country", "iso3", "region", "year"], value_vars=agegroup_cols)
    pop_df.rename(columns={"value": "population"}, inplace=True)

    def label_ages(age_str):
        if age_str.endswith("+"):
            return (int(age_str.replace("+", "")), None)
        else:
            return [int(s) for s in age_str.split("-")]

    ages_df = pd.DataFrame(
        [label_ages(age_str) for age_str in pop_df.variable],
        columns=("start_age", "end_age"),
    )
    pop_df = pop_df.join(ages_df)
    pop_df = pop_df.drop(columns="variable")

    # Get and append 2021 Bangladesh and Rohingya population
    pop_bgd = get_bangladesh_pop(BGD_POP)
    pop_df = pop_df.append(pop_bgd)

    pop_rohingya = get_rohingya_pop(ROHINGYA_POP)
    pop_df = pop_df.append(pop_rohingya)

    # Ensure all numbers are actually numbers
    numeric_cols = ["year", "start_age", "end_age", "population"]
    pop_df[numeric_cols] = pop_df[numeric_cols].apply(pd.to_numeric)
    pop_df = pop_df.sort_values(["iso3", "year", "start_age"])

    # Multiply population by 1000 to get actual numbers
    pop_df["population"] = 1e3 * pop_df["population"]

    return pop_df


def read_location_df():
    """
    Read UN country code mappings
    """
    location_path = os.path.join(POP_DIRPATH, "WPP2019_F01_LOCATIONS.xlsx")
    loc_df = pd.read_excel(
        pd.ExcelFile(location_path),
        header=16,
        index_col=0,
        sheet_name="Location",
    )
    rename_cols = {
        "Region, subregion, country or area*": "country",
        "ISO3 Alpha-code": "iso3",
        "Location code": "country_code",
    }
    loc_df.rename(columns=rename_cols, inplace=True)
    loc_df = loc_df[["country", "iso3", "country_code"]]
    # Drop all entries without a country code
    nan_value = float("NaN")
    loc_df.dropna(subset=["iso3"], inplace=True)
    loc_df["iso3"] = loc_df["iso3"].apply(lambda s: s.strip())
    loc_df["iso3"].replace("", nan_value, inplace=True)
    loc_df.dropna(subset=["iso3"], inplace=True)
    return loc_df


def get_bangladesh_pop(str: BGD_POP) -> pd.DataFrame:
    """Create a population dataframe for Bangladesh
    Args:
        str (BGD_POP): Path to new population file
    Returns:
        pd.DataFrame: A dataframe with 2021 population for Bangladesh
    """

    # Read and filter for row.
    bgd_df = pd.read_excel(BGD_POP, sheet_name="AgeGroup")
    dhk_df = bgd_df.loc[(bgd_df["District"] == "Dhaka") & (bgd_df["Sex"] == "Total")]
    bgd_df = bgd_df.loc[(bgd_df["Division"] == "Bangladesh") & (bgd_df["Sex"] == "Total")]

    dhk_df = extract_pop(dhk_df)
    bgd_df = extract_pop(bgd_df)

    dhk_df = add_pop_cols(dhk_df, "BGD", region="Dhaka district", year="2021")
    bgd_df = add_pop_cols(bgd_df, "BGD", year="2021")

    bgd_df = bgd_df.append(dhk_df)

    bgd_df = unpivot_df(bgd_df)
    return bgd_df


def extract_pop(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the population dataframe for Bangladesh and Dhaka city"""

    # Create 0-4 age group total and delete columns.
    df["0-4"] = df.loc[:, ["0 Year", "1 Year", "2 Years", "3 Years", "4 Years"]].sum(axis=1)
    df.drop(columns=["0 Year", "1 Year", "2 Years", "3 Years", "4 Years", "Sex"], inplace=True)

    # Fix column names and create additional ones.
    df.rename(columns=lambda x: x.lower().rstrip("years and abov").strip(), inplace=True)
    df.rename(columns={"divisi": "country", "district": "iso3", "80": "80-84"}, inplace=True)

    return df


def unpivot_df(df: pd.DataFrame) -> pd.DataFrame:
    """Unpivot the dataframe and splits the age brackets into columns"""

    # identify age group columns and unpivot
    agegroup_cols = [c for c in df.columns if "-" in c or c.endswith("+")]
    df = df.melt(
        id_vars=["country", "iso3", "region", "year"],
        value_vars=agegroup_cols,
        value_name="population",
    )

    df["population"] = df["population"] / 1000  # Divide to match UN pop data

    # Create start and end age groups.
    df["start_age"] = df["variable"].apply(lambda s: int(s.split("-")[0]))
    df["end_age"] = df["variable"].apply(lambda s: int(s.split("-")[1]))
    df.drop(columns="variable", inplace=True)
    return df


def add_pop_cols(
    df: pd.DataFrame, iso3: str, country: str = None, region: str = None, year: str = None
) -> pd.DataFrame:
    """Add missing population columns to the dataframe"""

    if "country" not in df.columns:
        df["country"] = country
    df["iso3"] = iso3
    df["region"] = region
    df["year"] = year

    return df


def get_rohingya_pop(ROHINGYA_POP: str) -> pd.DataFrame:
    """Reshape the Rohingya population dataframe"""

    df = pd.read_excel(ROHINGYA_POP, usecols=list(range(8, 20)), skiprows=6)
    df = df.tail(1)
    df["0-4"] = df.loc[
        :, ["below 1 year Infant", "Unnamed: 9", "between 1-4 year Children", "Unnamed: 11"]
    ].sum(axis=1)

    df["5-11"] = df.loc[:, ["between 5-11 year Children", "Unnamed: 13"]].sum(axis=1)
    df["12-17"] = df.loc[:, ["between 12-17 year Children", "Unnamed: 15"]].sum(axis=1)
    df["18-59"] = df.loc[:, ["between 18-59 year Adult", "Unnamed: 17"]].sum(axis=1)
    df["60-100"] = df.loc[:, ["60+ Elderly", "Unnamed: 19"]].sum(axis=1)

    df = add_pop_cols(df, "BGD", "Bangladesh", "FDMN", "2021")
    df.drop(
        columns=[
            "below 1 year Infant",
            "Unnamed: 9",
            "between 1-4 year Children",
            "Unnamed: 11",
            "between 5-11 year Children",
            "Unnamed: 13",
            "between 12-17 year Children",
            "Unnamed: 15",
            "between 18-59 year Adult",
            "Unnamed: 17",
            "60+ Elderly",
            "Unnamed: 19",
        ],
        inplace=True,
    )

    return unpivot_df(df)
