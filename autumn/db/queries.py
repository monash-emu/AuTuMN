"""
Functions for extracting data from a database.
"""
from typing import List

import numpy as np

from autumn.curve import scale_up_function
from .database import Database


def get_bcg_coverage(database, country_iso_code):
    """
    extract bcg coverage from inputs database

    :param database: sql database
        database containing the bcg data
    :param country_iso_code: string
        three letter ISO3 code for the country of interest
    :return: dict
        pandas data frame with columns years and one row containing the values of BCG coverage in that year
    """
    _bcg_coverage = database.db_query("BCG", conditions=["ISO_code='" + country_iso_code + "'"])
    _bcg_coverage = _bcg_coverage.filter(
        items=[column for column in _bcg_coverage.columns if column.isdigit()]
    )
    return {
        int(key): value / 1e2
        for key, value in zip(list(_bcg_coverage.columns), _bcg_coverage.loc[0, :])
        if value is not None
    }


def get_all_iso3_from_bcg(database):
    """
    check which iso3 country codes are available from the bcg database

    :param database: sql database
        the database containing the bcg data
    :return: list
        all the iso3 strings available from the bcg database
    """

    return database.db_query("bcg", column="ISO_code")["ISO_code"].tolist()


def get_crude_birth_rate(database, country_iso_code):
    """
    get the crude birth rate as a rate estimated by the un for a particular country

    :param database: sql database
        database containing the crude birth rate data from un
    :param country_iso_code: string
        three letter ISO3 code for the country of interest
    :return: dict
        keys for mid-point of year of interest with floats for the crude birth rate (per capita, not per 1,000)
    """

    # extract birth rates
    birth_rates = database.db_query(
        "crude_birth_rate_mapped", conditions=["iso3='" + country_iso_code + "'"]
    )

    # find the keys with a - in them to indicate a time range and add 2.5 on to the starting value to get mid-point
    return {
        float(key[: key.find("-")]) + 2.5: float(value) / 1e3
        for key, value in zip(list(birth_rates.columns), birth_rates.loc[0, :])
        if "-" in key
    }


def extract_demo_data(_input_database, data_type, country_iso_code):
    """
    get and format demographic data from the input databases originally derived from the un sources
    note that the number of period that data are provided for differs for total population and absolute deaths

    :param _input_database: sql database
        the master inputs database
    :param data_type: str
        the database type of interest
    :param country_iso_code: str
        the three digit iso3 code for the country of interest
    :return: pandas dataframe
        cleaned pandas dataframe ready for use in demographic calculations
    """

    # get the appropriate data type from the un-derived databases
    demo_data_frame = _input_database.db_query(
        data_type, conditions=["iso3='" + country_iso_code + "'"]
    )

    # rename columns, including adding a hyphen to the last age group to make it behave like the others age groups
    demo_data_frame.rename(
        columns={"95+": "95-", "Reference date (as of 1 July)": "Period"}, inplace=True
    )

    # retain only the relevant columns
    columns_to_keep = [column for column in demo_data_frame.columns if "-" in column]
    columns_to_keep.append("Period")
    demo_data_frame = demo_data_frame.loc[:, columns_to_keep]

    # rename the columns to make them integers
    demo_data_frame.columns = [
        int(column[: column.find("-")]) if "-" in column else column
        for column in list(demo_data_frame.columns)
    ]

    # change the year data for the period to numeric type
    demo_data_frame["Period"] = demo_data_frame["Period"].apply(
        lambda x: str(x)[: str(x).find("-")] if "-" in str(x) else str(x)
    )

    # return final version
    return demo_data_frame


def clean_age_breakpoints(breakpoints):
    bs = sorted(breakpoints)
    return bs if 0 in breakpoints else [0, *bs]


def find_death_rates(_input_database, country_iso_code):
    """
    find death rates by reported age bracket from database populated from un data

    :param _input_database: sql database
        the inputs database
    :param country_iso_code: str
        the three digit iso3 code for the country of interest
    :return: pandas dataframe:
        mortality rates by age bracket
        mortality_years: list
        values of the mid-points of the years for which mortality is estimated
    """

    # get necessary data from database
    absolute_death_data = extract_demo_data(
        _input_database, "absolute_deaths_mapped", country_iso_code
    )
    total_population_data = extract_demo_data(
        _input_database, "total_population_mapped", country_iso_code
    )

    # cut off last row of population data because it goes out five years longer
    total_population_data = total_population_data.loc[: absolute_death_data.shape[0] - 1, :]

    # cut off last column of both data frames because they include the years, but retain the data as a list
    mortality_years = [float(i) + 2.5 for i in list(total_population_data.loc[:, "Period"])]
    total_population_data = total_population_data.iloc[:, : total_population_data.shape[1] - 1]
    absolute_death_data = absolute_death_data.iloc[:, : absolute_death_data.shape[1] - 1]

    # make sure all floats, as seem to have become str somewhere
    absolute_death_data = absolute_death_data.astype(float)
    total_population_data = total_population_data.astype(float)

    # replace NaN and inf values with zeros
    death_rates = absolute_death_data / total_population_data / 5.0
    for col in death_rates.columns:
        death_rates[col] = death_rates[col].fillna(0)
        death_rates[col] = death_rates[col].replace(np.inf, 0.0)

    # divide through by population and by five to allow for the mortality data being aggregated over five year periods
    return death_rates, mortality_years


def find_age_weights(
    age_breakpoints, demo_data, arbitrary_upper_age=1e2, break_width=5.0, normalise=True
):
    """
    find the weightings to assign to the various components of the data from the age breakpoints planned to be used
    in the model

    :param age_breakpoints: list
        numeric values for the breakpoints of the age brackets
    :param demo_data: pandas dataframe
        the demographic data extracted from the database into pandas format
    :param arbitrary_upper_age: float
        arbitrary upper value to consider for the highest age bracket
    :param break_width: float
        difference between the lower and upper values of the age brackets in the data
    :return: dict
        keys age breakpoints, values list of the weightings to assign to the data age categories
    """
    weightings_dict = {}

    # cycle through each age bracket/category
    for n_breakpoint in range(len(age_breakpoints)):

        lower_value = age_breakpoints[n_breakpoint]
        upper_value = (
            arbitrary_upper_age
            if n_breakpoint == len(age_breakpoints) - 1
            else age_breakpoints[n_breakpoint + 1]
        )

        # initialise weights to one and then subtract parts of bracket that are excluded
        weightings = [1.0] * len(demo_data.columns)

        # cycle through the breakpoints of the data on inner loop
        for n_data_break, data_breakpoints in enumerate(demo_data.columns):
            data_lower = data_breakpoints
            data_upper = data_breakpoints + break_width

            # first consider the lower value of the age bracket and how much of the data it excludes
            if data_upper <= lower_value:
                weightings[n_data_break] -= 1.0
            elif data_lower < lower_value < data_upper:
                weightings[n_data_break] -= 1.0 - (data_upper - lower_value) / break_width

            # then consider the upper value of the age bracket and how much of the data it excludes
            if data_lower < upper_value < data_upper:
                weightings[n_data_break] -= 1.0 - (upper_value - data_lower) / break_width
            elif upper_value <= data_lower:
                weightings[n_data_break] -= 1.0

        # normalise the values
        if normalise:
            weightings = [weight / sum(weightings) for weight in weightings]
        weightings_dict[age_breakpoints[n_breakpoint]] = weightings
    return weightings_dict


def find_age_specific_death_rates(input_database, age_breakpoints, country_iso_code):
    """
    find non-tb-related death rates from un data that are specific to the age groups requested for the model regardless
    of the age brackets for which data are available

    :param age_breakpoints: list
        integers for the age breakpoints being used in the model
    :param country_iso_code: str
        the three digit iso3 code for the country of interest
    :return: dict
        keys the age breakpoints, values lists for the death rates with time
    """
    age_breakpoints = clean_age_breakpoints(age_breakpoints)

    # gather up the death rates with the brackets from the data
    death_rates, years = find_death_rates(input_database, country_iso_code)

    # find the weightings to each age group in the data from the requested brackets
    age_weights = find_age_weights(age_breakpoints, death_rates)

    # calculate the list of values for the weighted death rates for each modelled age category
    age_death_rates = {}
    for age_break in age_breakpoints:
        age_death_rates[age_break] = [0.0] * death_rates.shape[0]
        for year in death_rates.index:
            age_death_rates[age_break][year] = sum(
                [
                    death_rate * weight
                    for death_rate, weight in zip(
                        list(death_rates.iloc[year]), age_weights[age_break]
                    )
                ]
            )
    return age_death_rates, years


def find_population_by_agegroup(
    input_database: Database, age_breakpoints: List[float], country_iso_code: str
):
    """
    Find population for age brackets.
    Returns a dict of lists.
    """
    age_breakpoints = clean_age_breakpoints(age_breakpoints)
    data_type = "total_population_mapped"
    total_population_data = extract_demo_data(input_database, data_type, country_iso_code)
    years = list(total_population_data["Period"])
    populations = total_population_data.drop(["Period"], axis=1)
    age_weights = find_age_weights(age_breakpoints, populations, normalise=False)
    age_populations = {}
    for age_break in age_breakpoints:
        age_populations[age_break] = [0.0] * populations.shape[0]
        for i_year, year in enumerate(years):
            age_populations[age_break][i_year] = sum(
                [
                    float(pop) * weight
                    for pop, weight in zip(list(populations.iloc[i_year]), age_weights[age_break])
                ]
            )

    return age_populations, years


def get_pop_mortality_functions(
    input_database,
    age_breaks,
    country_iso_code,
    emigration_value=0.0,
    emigration_start_time=1980.0,
):
    """
    use the mortality rate data that can be obtained from find_age_specific_death_rates to fit time-variant mortality
        functions for each age group being implemented in the model

    :param age_breaks: list
        starting ages for each of the age groups
    :param country_iso_code: str
        the three digit iso3 code for the country of interest
    :param emigration_value: float
        an extra rate of migration to add on to the population-wide mortality rates to simulate net emigration
    :param emigration_start_time: float
        the point from which the additional net emigration commences
    :return: dict
        keys age breakpoints, values mortality functions
    """
    age_death_dict, data_years = find_age_specific_death_rates(
        input_database, age_breaks, country_iso_code
    )

    # add an extra fixed value after a particular time point for each mortality estimate
    for age_group in age_death_dict:
        for i_year in range(len(age_death_dict[age_group])):
            if data_years[i_year] > emigration_start_time:
                age_death_dict[age_group][i_year] += emigration_value

    # fit the curve functions to the aggregate data of mortality and net emigration
    return {
        age_group: scale_up_function(
            data_years, age_death_dict[age_group], smoothness=0.2, method=5
        )
        for age_group in age_death_dict
    }


def get_iso3_from_country_name(input_database, country_name):
    """
    Return the iso3 code matching with a given country name using the bcg table
    """
    if "Lao " in country_name:
        return "LAO"

    iso3_list = input_database.db_query(
        "bcg", column="ISO_code", conditions=["Cname='" + country_name + "'"]
    )["ISO_code"].tolist()

    if len(iso3_list) > 0:
        return iso3_list[0]
    else:
        backup_iso3_codes = {
            "Andorra": "AND",
            "Antigua and Barbuda": "ATG",
            "Australia": "AUS",
            "Bahamas": "BHS",
            "Bahrain": "BHR",
            "Belgium": "BEL",
            "Bolivia (Plurinational State of": "BOL",
            "Canada": "CAN",
            "Congo": "COG",
            "Cyprus": "CYP",
            "Dominican Republic": "DOM",
            "Germany": "DEU",
            "Hong Kong SAR, China": "HKG",
            "Iceland": "ISL",
            "Lebanon": "LBN",
            "Luxembourg": "LUX",
            "Netherlands": "NLD",
            "New Zealand": "NZL",
            "Niger": "NER",
            "Philippines": "PHL",
            "Republic of Korea": "KOR",
            "Russian Federation": "RUS",
            "Sao Tome and Principe ": "STP",
            "Spain": "ESP",
            "Suriname": "SUR",
            "Switzerland": "CHE",
            "Syrian Arab Republic": "SYR",
            "Taiwan": "TWN",
            "TFYR of Macedonia": "MKD",
            "United Arab Emirates": "ARE",
            "United Kingdom of Great Britain": "GBR",
            "United States of America": "USA",
            "Venezuela (Bolivarian Republic ": "VEN",
        }

        if country_name in backup_iso3_codes:
            return backup_iso3_codes[country_name]
        else:
            return None


def get_country_name_from_iso3(input_database, iso3):
    """
    Return the country name matching with a given iso3 code using the bcg table
    """
    country_list = input_database.db_query(
        "bcg", column="Cname", conditions=["ISO_code='" + iso3 + "'"]
    )["Cname"].tolist()

    return None if len(country_list) == 0 else country_list[0]
